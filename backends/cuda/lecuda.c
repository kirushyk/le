/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <cublas.h>
#include <cublas_v2.h>
#include "lecuda.h"
#include <assert.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>

LeTensor *
le_cuda_matrix_new_product(const LeTensor *a, bool transpose_a, const LeTensor *b, bool transpose_b)
{
    assert(a->device_type == LE_DEVICE_TYPE_CUDA);
    assert(b->device_type == LE_DEVICE_TYPE_CUDA);
    
    assert(a->element_type == LE_TYPE_FLOAT32);
    assert(b->element_type == LE_TYPE_FLOAT32);
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    assert(le_tensor_contiguous(a));
    assert(le_tensor_contiguous(b));
    
    unsigned size_a = transpose_a ? a->shape->sizes[0] : a->shape->sizes[1];
    unsigned size_b = transpose_b ? b->shape->sizes[1] : b->shape->sizes[0];
    assert(size_a == size_b);
    
    unsigned c_height = transpose_a ? a->shape->sizes[1] : a->shape->sizes[0];
    unsigned c_width = transpose_b ? b->shape->sizes[0] : b->shape->sizes[1];

    LeTensor *c = malloc(sizeof(struct LeTensor));
    c->device_type = LE_DEVICE_TYPE_CUDA;
    c->element_type = a->element_type;
    c->shape = le_shape_new(2, c_height, c_width);
    c->stride = le_shape_get_size(c->shape, -1);
    c->owns_data = true;
    size_t data_size = le_shape_get_elements_count(c->shape) * le_type_size(c->element_type);
    
    cudaError_t cuda_res;
    cuda_res = cudaMalloc((void**)&c->data, data_size);
    assert(cuda_res == cudaSuccess);
    cuda_res = cudaMemset(c->data, 0, data_size);
    assert(cuda_res == cudaSuccess);

    float alpha = 1.0f, beta = 0.0f;

    cublasHandle_t handle;
    cublasStatus_t cublas_status;
    cublas_status = cublasCreate(&handle);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cublasSgemm(handle,
        transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
        transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
        c_width,
        c_height,
        size_a,
        &alpha,
        b->data, /// B Transposed
        b->shape->sizes[1],
        a->data, /// A Transposed
        a->shape->sizes[1],
        &beta,
        c->data,
        c_width /// C Transposed
    );
    cublasDestroy(handle);
    
    return c;
}

extern void hadamard_wrapper(float *a, float *b, int l);

void 
le_cuda_tensor_mul_tensor(LeTensor *self, const LeTensor *b)
{
    assert(self->device_type == LE_DEVICE_TYPE_CUDA);
    assert(b->device_type == LE_DEVICE_TYPE_CUDA);
    assert(self->element_type == LE_TYPE_FLOAT32);
    assert(b->element_type == LE_TYPE_FLOAT32);
    assert(self->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    assert(le_tensor_contiguous(self));
    assert(le_tensor_contiguous(b));
    hadamard_wrapper(self->data, b->data, le_shape_get_elements_count(self->shape));
    cudaDeviceSynchronize();
}

extern void sigmoid_wrapper(float *a, int l);

void
le_cuda_tensor_apply_sigmoid(LeTensor *self)
{
    assert(self->device_type == LE_DEVICE_TYPE_CUDA);
    assert(self->element_type == LE_TYPE_FLOAT32);
    assert(le_tensor_contiguous(self));
    sigmoid_wrapper(self->data, le_shape_get_elements_count(self->shape));
    cudaDeviceSynchronize();
}

extern void sigmoid_prime_wrapper(float *a, int l);

void
le_cuda_tensor_apply_sigmoid_prime(LeTensor *self)
{
    assert(self->device_type == LE_DEVICE_TYPE_CUDA);
    assert(self->element_type == LE_TYPE_FLOAT32);
    assert(le_tensor_contiguous(self));
    sigmoid_prime_wrapper(self->data, le_shape_get_elements_count(self->shape));
    cudaDeviceSynchronize();
}

LeTensor *
le_tensor_to_cuda(const LeTensor *cpu_tensor)
{
    assert(cpu_tensor);
    assert(le_tensor_contiguous(cpu_tensor));
    assert(cpu_tensor->device_type == LE_DEVICE_TYPE_CPU);
    
    LeTensor *tensor = malloc(sizeof(struct LeTensor));
    tensor->device_type = LE_DEVICE_TYPE_CUDA;
    tensor->element_type = cpu_tensor->element_type;
    tensor->shape = le_shape_copy(cpu_tensor->shape);
    tensor->stride = le_shape_get_size(cpu_tensor->shape, -1);
    tensor->owns_data = true;
    size_t data_size = le_shape_get_elements_count(tensor->shape) * le_type_size(tensor->element_type);
    
    cudaError_t cuda_res;
    cuda_res = cudaMalloc((void**)&tensor->data, data_size);
    assert(cuda_res == cudaSuccess);
    cublasStatus_t cublas_status;
    cublas_status = cublasSetMatrix(cpu_tensor->shape->sizes[1], cpu_tensor->shape->sizes[0], sizeof(float), cpu_tensor->data, cpu_tensor->shape->sizes[1], tensor->data, tensor->shape->sizes[1]);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    return tensor;
}
   
LeTensor *
le_cuda_tensor_to_cpu(const LeTensor *cuda_tensor)
{
    assert(cuda_tensor);
    assert(le_tensor_contiguous(cuda_tensor));
    assert(cuda_tensor->device_type == LE_DEVICE_TYPE_CUDA);
    
    LeTensor *tensor = malloc(sizeof(struct LeTensor));
    tensor->device_type = LE_DEVICE_TYPE_CPU;
    tensor->element_type = cuda_tensor->element_type;
    tensor->shape = le_shape_copy(cuda_tensor->shape);
    tensor->stride = le_shape_get_size(cuda_tensor->shape, -1);
    tensor->owns_data = true;
    size_t data_size = le_shape_get_elements_count(tensor->shape) * le_type_size(tensor->element_type);

    tensor->data = malloc(data_size);
    cublasStatus_t cublas_status;
    cublas_status = cublasGetMatrix(cuda_tensor->shape->sizes[1], cuda_tensor->shape->sizes[0], sizeof(float), cuda_tensor->data, cuda_tensor->shape->sizes[1], tensor->data, tensor->shape->sizes[1]);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    
    return tensor;
}

void *
le_cuda_data_copy(void *data, size_t bytes)
{
    void *dataCopy;
    cudaError_t cuda_res;
    cuda_res = cudaMalloc((void**)&dataCopy, bytes);
    assert(cuda_res == cudaSuccess);
    cudaMemcpy(dataCopy, data, bytes, cudaMemcpyDeviceToDevice);
    return dataCopy;
}

void
le_cuda_data_free(void *data)
{
    if (data)
    {
        cudaFree(data);
    }
}
