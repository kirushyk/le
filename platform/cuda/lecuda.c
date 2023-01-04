/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <cublas.h>
#include <cublas_v2.h>
#include "lecuda.h"
#include <assert.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>

LeTensor *
le_cuda_matrix_new_product(const LeTensor *a, bool transpose_a, const LeTensor *b, bool transpose_b)
{
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

    cudaError_t cuda_res;
    cublasStatus_t cublas_status;
    cublasHandle_t handle;

    cublas_status = cublasCreate(&handle);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    float *dev_at = NULL, *dev_bt = NULL, *dev_ct = NULL;

    cuda_res = cudaMalloc((void**)&dev_at, a->shape->sizes[1] * a->shape->sizes[0] * sizeof(float));
    assert(cuda_res == cudaSuccess);
    
    cublas_status = cublasSetMatrix(a->shape->sizes[1], a->shape->sizes[0], sizeof(float), a->data, a->shape->sizes[1], dev_at, a->shape->sizes[1]);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    cuda_res = cudaMalloc((void**)&dev_bt, b->shape->sizes[1] * b->shape->sizes[0] * sizeof(float));
    assert(cuda_res == cudaSuccess);

    cublas_status = cublasSetMatrix(b->shape->sizes[1], b->shape->sizes[0], sizeof(float), b->data, b->shape->sizes[1], dev_bt, b->shape->sizes[1]);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    
    cuda_res = cudaMalloc((void**)&dev_ct, c_height * c_width * sizeof(float));
    assert(cuda_res == cudaSuccess);
    cuda_res = cudaMemset(dev_ct, 0, c_height * c_width * sizeof(float));
    assert(cuda_res == cudaSuccess);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle,
        transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
        transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
        c_width,
        c_height,
        size_a,
        &alpha,
        dev_bt,
        transpose_b ? b->shape->sizes[0] : b->shape->sizes[1],
        dev_at,
        transpose_a ? a->shape->sizes[0] : a->shape->sizes[1],
        &beta,
        dev_ct,
        c_height
    );
    
    LeTensor *c = le_matrix_new_uninitialized(LE_TYPE_FLOAT32, c_height, c_width);
    cublas_status = cublasGetMatrix(c->shape->sizes[1], c->shape->sizes[0], sizeof(float), dev_ct, c->shape->sizes[1], c->data, c->shape->sizes[1]);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
        
    cudaFree(dev_ct);
    cudaFree(dev_bt);
    cudaFree(dev_at);
    cublasDestroy(handle);

    return c;
}
