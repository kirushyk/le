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
    
    LeTensor *c = le_matrix_new_uninitialized(LE_TYPE_FLOAT32, c_height, c_width);

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    assert(stat == CUBLAS_STATUS_SUCCESS);

    float *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;

    cudaStat = cudaMalloc((void**)&dev_a, a->shape->sizes[0] * a->shape->sizes[1] * sizeof(float));
    assert(cudaStat == cudaSuccess);
    
    stat = cublasSetMatrix(a->shape->sizes[0], a->shape->sizes[1], sizeof(float), a->data, a->shape->sizes[0], dev_a, a->shape->sizes[0]);
    assert(stat == CUBLAS_STATUS_SUCCESS);

    cudaStat = cudaMalloc((void**)&dev_b, b->shape->sizes[0] * b->shape->sizes[1] * sizeof(float));
    assert(cudaStat == cudaSuccess);

    stat = cublasSetMatrix(b->shape->sizes[0], b->shape->sizes[1], sizeof(float), b->data, b->shape->sizes[0], dev_b, b->shape->sizes[0]);
    assert(stat == CUBLAS_STATUS_SUCCESS);
    
    cudaStat = cudaMalloc((void**)&dev_c, c_height * c_width * sizeof(float));
    assert(cudaStat == cudaSuccess);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle,
        transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
        transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
        c_height, c_width, size_a,
        &alpha,
        dev_a, a->shape->sizes[1],
        dev_b, b->shape->sizes[1],
        &beta,
        dev_c, c->shape->sizes[1]);

    stat = cublasGetMatrix(c->shape->sizes[0], c->shape->sizes[1], sizeof(float), dev_c, c->shape->sizes[0], c->data, c->shape->sizes[0]);
    assert(stat == CUBLAS_STATUS_SUCCESS);
        
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
    cublasDestroy(handle);

    return c;
}
