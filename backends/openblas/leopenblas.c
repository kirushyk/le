/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leopenblas.h"
#include <assert.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>
#include <cblas.h>

LeTensor *
le_openblas_matrix_new_product(const LeTensor *a, bool transpose_a, const LeTensor *b, bool transpose_b)
{
    /// @todo: Take stride into account
    assert(a->element_type == LE_TYPE_F32);
    assert(b->element_type == LE_TYPE_F32);
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    assert(le_tensor_contiguous(a));
    assert(le_tensor_contiguous(b));
    
    unsigned size_a = transpose_a ? a->shape->sizes[0] : a->shape->sizes[1];
    unsigned size_b = transpose_b ? b->shape->sizes[1] : b->shape->sizes[0];
    assert(size_a == size_b);
    
    unsigned c_height = transpose_a ? a->shape->sizes[1] : a->shape->sizes[0];
    unsigned c_width = transpose_b ? b->shape->sizes[0] : b->shape->sizes[1];
    
    LeTensor *c = le_matrix_new_uninitialized(LE_TYPE_F32, c_height, c_width);
    
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

gfloat
le_openblas_dot_product(const LeTensor *a, const LeTensor *b)
{
    assert(a->element_type == LE_TYPE_F32);
    assert(b->element_type == LE_TYPE_F32);
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    /** @todo: Test results against transposed a multiplied by b */
    assert(a->shape->sizes[0] == b->shape->sizes[0]);
    assert(a->shape->sizes[1] == 1);
    assert(b->shape->sizes[1] == 1);
    
    return cblas_sdot(a->shape->sizes[0], a->data, a->stride, b->data, b->stride);
}
