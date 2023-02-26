/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leaccelerate.h"
#include <assert.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>
#include <Accelerate/Accelerate.h>

LeTensor *
le_accelerate_matrix_new_product(const LeTensor *a, bool transpose_a, const LeTensor *b, bool transpose_b)
{
    /// @todo: Take stride into account
    assert(a->element_type == LE_TYPE_FLOAT32);
    assert(b->element_type == LE_TYPE_FLOAT32);
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    // assert(le_tensor_contiguous(a));
    // assert(le_tensor_contiguous(b));
    
    unsigned size_a = transpose_a ? a->shape->sizes[0] : a->shape->sizes[1];
    unsigned size_b = transpose_b ? b->shape->sizes[1] : b->shape->sizes[0];
    assert(size_a == size_b);
    
    unsigned c_height = transpose_a ? a->shape->sizes[1] : a->shape->sizes[0];
    unsigned c_width = transpose_b ? b->shape->sizes[0] : b->shape->sizes[1];
    
    LeTensor *c = le_matrix_new_uninitialized(LE_TYPE_FLOAT32, c_height, c_width);
    
    cblas_sgemm(CblasRowMajor,
                transpose_a ? CblasTrans : CblasNoTrans,
                transpose_b ? CblasTrans : CblasNoTrans,
                c_height, c_width, size_a,
                1.0f,
                a->data, a->stride,
                b->data, b->stride,
                0.0f,
                c->data, c->stride);
    
    return c;
}

void
le_accelerate_tensor_apply_sigmoid(LeTensor *tensor)
{
    /// @todo: Take stride into account
    assert(tensor);
    assert(tensor->element_type == LE_TYPE_FLOAT32);
        
    int n = le_shape_get_elements_count(tensor->shape);
    vDSP_vneg(tensor->data, 1, tensor->data, 1, n);
    vvexpf(tensor->data, tensor->data, &n);
    float one = 1.0f;
    vDSP_vsadd(tensor->data, 1, &one, tensor->data, 1, n);
    vDSP_svdiv(&one, tensor->data, 1, tensor->data, 1, n);
}

void
le_accelerate_tensor_apply_sigmoid_prime(LeTensor *tensor)
{
    /// @todo: Take stride into account
    assert(tensor);
    assert(tensor->element_type == LE_TYPE_FLOAT32);
        
    /// @note: I do not want to compute n twice so I will not call le_accelerate_tensor_apply_sigmoid
    int n = le_shape_get_elements_count(tensor->shape);
    vDSP_vneg(tensor->data, 1, tensor->data, 1, n);
    vvexpf(tensor->data, tensor->data, &n);
    float one = 1.0f;

    vDSP_vsadd(tensor->data, 1, &one, tensor->data, 1, n);
    vDSP_svdiv(&one, tensor->data, 1, tensor->data, 1, n);
    
    for (unsigned i = 0; i < n; i++)
    {
        float sigmoid = ((float *)tensor->data)[i];
        ((float *)tensor->data)[i] = sigmoid * (1 - sigmoid);
    }
}

float
le_accelerate_rbf(const LeTensor *a, const LeTensor *b, float sigma)
{
    assert(a->element_type == LE_TYPE_FLOAT32);
    assert(b->element_type == LE_TYPE_FLOAT32);
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    /** @todo: Test results against transposed a multiplied by b */
    assert(a->shape->sizes[0] == b->shape->sizes[0]);
    assert(a->shape->sizes[1] == 1);
    assert(b->shape->sizes[1] == 1);
    
    float *c = malloc(sizeof(float) * a->shape->sizes[0]);
    
    float result;
    vDSP_vsub(a->data, a->stride, b->data, b->stride, c, 1, a->shape->sizes[0]);
    vDSP_svesq(c, 1, &result, a->shape->sizes[0]);

    free(c);
    
    return expf(-result / (2.0f * sigma * sigma));
}

float
le_accelerate_dot_product(const LeTensor *a, const LeTensor *b)
{
    assert(a->element_type == LE_TYPE_FLOAT32);
    assert(b->element_type == LE_TYPE_FLOAT32);
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    /** @todo: Test results against transposed a multiplied by b */
    assert(a->shape->sizes[0] == b->shape->sizes[0]);
    assert(a->shape->sizes[1] == 1);
    assert(b->shape->sizes[1] == 1);
    
    return cblas_sdot(a->shape->sizes[0], a->data, a->stride, b->data, b->stride);
}
