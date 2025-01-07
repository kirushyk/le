/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lecuda.h"
#include <assert.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>

LeTensor *
le_cuda_matrix_new_product (const LeTensor *a, bool transpose_a, const LeTensor *b, bool transpose_b)
{
  g_assert_cmpint (a->device_type, ==, LE_DEVICE_TYPE_CUDA);
  g_assert_cmpint (b->device_type, ==, LE_DEVICE_TYPE_CUDA);

  g_assert_cmpint (a->element_type, ==, LE_TYPE_FLOAT32);
  g_assert_cmpint (b->element_type, ==, LE_TYPE_FLOAT32);
  g_assert_cmpint (a->shape->num_dimensions, ==, 2);
  g_assert_cmpint (b->shape->num_dimensions, ==, 2);
  g_assert_true (le_tensor_contiguous (a));
  g_assert_true (le_tensor_contiguous (b));

  gsize size_a = transpose_a ? a->shape->sizes[0] : a->shape->sizes[1];
  gsize size_b = transpose_b ? b->shape->sizes[1] : b->shape->sizes[0];
  g_assert_cmpint (size_a, ==, size_b);

  gsize c_height = transpose_a ? a->shape->sizes[1] : a->shape->sizes[0];
  gsize c_width = transpose_b ? b->shape->sizes[0] : b->shape->sizes[1];

  LeTensor *c = g_new0 (LeTensor, 1);
  c->device_type = LE_DEVICE_TYPE_CUDA;
  c->element_type = a->element_type;
  c->shape = le_shape_new (2, c_height, c_width);
  c->stride = le_shape_get_size (c->shape, -1);
  c->owns_data = true;
  gsize data_size = le_shape_get_elements_count (c->shape) * le_type_size (c->element_type);

  cudaError_t cuda_res;
  cuda_res = cudaMalloc ((void **)&c->data, data_size);
  g_assert_cmpint (cuda_res, ==, cudaSuccess);
  cuda_res = cudaMemset (c->data, 0, data_size);
  g_assert_cmpint (cuda_res, ==, cudaSuccess);

  gfloat alpha = 1.0f, beta = 0.0f;

  cublasHandle_t handle;
  cublasStatus_t cublas_status;
  cublas_status = cublasCreate (&handle);
  assert (cublas_status == CUBLAS_STATUS_SUCCESS);
  cublasSgemm (handle, transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N, transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N, c_width,
      c_height, size_a, &alpha,
      b->data, /// B Transposed
      b->shape->sizes[1],
      a->data, /// A Transposed
      a->shape->sizes[1], &beta, c->data,
      c_width /// C Transposed
  );
  cublasDestroy (handle);

  return c;
}

extern void hadamard_wrapper (gfloat *a, gfloat *b, int l);

void
le_cuda_tensor_mul_tensor (LeTensor *self, const LeTensor *b)
{
  g_assert_cmpint (self->device_type, ==, LE_DEVICE_TYPE_CUDA);
  g_assert_cmpint (b->device_type, ==, LE_DEVICE_TYPE_CUDA);
  g_assert_cmpint (self->element_type, ==, LE_TYPE_FLOAT32);
  g_assert_cmpint (b->element_type, ==, LE_TYPE_FLOAT32);
  g_assert_cmpint (self->shape->num_dimensions, ==, 2);
  g_assert_cmpint (b->shape->num_dimensions, ==, 2);
  g_assert_true (le_tensor_contiguous (self));
  g_assert_true (le_tensor_contiguous (b));
  hadamard_wrapper (self->data, b->data, le_shape_get_elements_count (self->shape));
  cudaDeviceSynchronize ();
}

extern void sigmoid_wrapper (gfloat *a, int l);

void
le_cuda_tensor_apply_sigmoid (LeTensor *self)
{
  g_assert_cmpint (self->device_type, ==, LE_DEVICE_TYPE_CUDA);
  g_assert_cmpint (self->element_type, ==, LE_TYPE_FLOAT32);
  g_assert_true (le_tensor_contiguous (self));
  sigmoid_wrapper (self->data, le_shape_get_elements_count (self->shape));
  cudaDeviceSynchronize ();
}

extern void sigmoid_prime_wrapper (gfloat *a, int l);

void
le_cuda_tensor_apply_sigmoid_prime (LeTensor *self)
{
  g_assert_cmpint (self->device_type, ==, LE_DEVICE_TYPE_CUDA);
  g_assert_cmpint (self->element_type, ==, LE_TYPE_FLOAT32);
  g_assert_true (le_tensor_contiguous (self));
  sigmoid_prime_wrapper (self->data, le_shape_get_elements_count (self->shape));
  cudaDeviceSynchronize ();
}

LeTensor *
le_tensor_to_cuda (const LeTensor *cpu_tensor)
{
  g_assert_nonnull (cpu_tensor);
  g_assert_true (le_tensor_contiguous (cpu_tensor));
  g_assert_cmpint (cpu_tensor->device_type, ==, LE_DEVICE_TYPE_CPU);

  LeTensor *tensor = g_new0 (LeTensor, 1);
  tensor->device_type = LE_DEVICE_TYPE_CUDA;
  tensor->element_type = cpu_tensor->element_type;
  tensor->shape = le_shape_copy (cpu_tensor->shape);
  tensor->stride = le_shape_get_size (cpu_tensor->shape, -1);
  tensor->owns_data = true;
  gsize data_size = le_shape_get_elements_count (tensor->shape) * le_type_size (tensor->element_type);

  cudaError_t cuda_res;
  cuda_res = cudaMalloc ((void **)&tensor->data, data_size);
  assert (cuda_res == cudaSuccess);
  cublasStatus_t cublas_status;
  cublas_status = cublasSetMatrix (cpu_tensor->shape->sizes[1], cpu_tensor->shape->sizes[0], sizeof (gfloat),
      cpu_tensor->data, cpu_tensor->shape->sizes[1], tensor->data, tensor->shape->sizes[1]);
  assert (cublas_status == CUBLAS_STATUS_SUCCESS);

  return tensor;
}

LeTensor *
le_cuda_tensor_to_cpu (const LeTensor *cuda_tensor)
{
  g_assert_nonnull (cuda_tensor);
  g_assert_true (le_tensor_contiguous (cuda_tensor));
  g_assert_cmpint (cuda_tensor->device_type, ==, LE_DEVICE_TYPE_CUDA);

  LeTensor *tensor = g_new0 (LeTensor, 1);
  tensor->device_type = LE_DEVICE_TYPE_CPU;
  tensor->element_type = cuda_tensor->element_type;
  tensor->shape = le_shape_copy (cuda_tensor->shape);
  tensor->stride = le_shape_get_size (cuda_tensor->shape, -1);
  tensor->owns_data = true;
  gsize data_size = le_shape_get_elements_count (tensor->shape) * le_type_size (tensor->element_type);

  tensor->data = g_malloc (data_size);
  cublasStatus_t cublas_status;
  cublas_status = cublasGetMatrix (cuda_tensor->shape->sizes[1], cuda_tensor->shape->sizes[0], sizeof (gfloat),
      cuda_tensor->data, cuda_tensor->shape->sizes[1], tensor->data, tensor->shape->sizes[1]);
  assert (cublas_status == CUBLAS_STATUS_SUCCESS);

  return tensor;
}

void *
le_cuda_data_copy (void *data, gsize bytes)
{
  void *dataCopy;
  cudaError_t cuda_res = cudaMalloc ((void **)&dataCopy, bytes);
  g_assert_cmpint (cuda_res, ==, cudaSuccess);
  cudaMemcpy (dataCopy, data, bytes, cudaMemcpyDeviceToDevice);
  return dataCopy;
}

void
le_cuda_data_free (void *data)
{
  if (data) {
    cudaFree (data);
  }
}
