/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leaccelerate.h"
#include <assert.h>
#include <le/le.h>
#include <le/letensor-imp.h>
#include <Accelerate/Accelerate.h>

LeTensor *
le_accelerate_matrix_new_product(LeTensor *a, bool transpose_a, LeTensor *b, bool transpose_b)
{
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    
    unsigned size_a = transpose_a ? a->shape->sizes[0] : a->shape->sizes[1];
    unsigned size_b = transpose_b ? b->shape->sizes[1] : b->shape->sizes[0];
    
    assert(size_a == size_b);
    
    unsigned c_height = transpose_a ? a->shape->sizes[1] : a->shape->sizes[0];
    unsigned c_width = transpose_b ? b->shape->sizes[0] : b->shape->sizes[1];
    
    LeTensor *c = le_matrix_new_uninitialized(c_height, c_width);
    
    cblas_sgemm(CblasRowMajor,
                transpose_a ? CblasTrans : CblasNoTrans,
                transpose_b ? CblasTrans : CblasNoTrans,
                c_height, c_width, size_a,
                1.0f,
                a->data, a->shape->sizes[1],
                b->data, b->shape->sizes[1],
                0.0f,
                c->data, c->shape->sizes[1]);
    
    return c;
}

void
le_accelerate_tensor_apply_sigmoid(LeTensor *tensor)
{
    assert(tensor);
    assert(tensor->element_type == LE_TYPE_FLOAT32);
        
    int n = le_shape_get_elements_count(tensor->shape);
    vDSP_vneg(tensor->data, 1, tensor->data, 1, n);
    vvexpf(tensor->data, tensor->data, &n);
    float one = 1.0f;
    vDSP_vsadd(tensor->data, 1, &one, tensor->data, 1, n);
    vDSP_svdiv(&one, tensor->data, 1, tensor->data, 1, n);
}
