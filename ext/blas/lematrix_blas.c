/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lematrix_blas.h"
#include <assert.h>
#include <le/le.h>
#include <le/letensor-imp.h>
#include <Accelerate/Accelerate.h>

LeTensor *
le_blas_matrix_new_product(LeTensor *a, LeTensor *b)
{
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    assert(a->shape->sizes[1] == b->shape->sizes[0]);
    
    LeTensor *c = le_matrix_new_uninitialized(a->shape->sizes[0], b->shape->sizes[1]);
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                a->shape->sizes[0], b->shape->sizes[1], a->shape->sizes[1],
                1.0f,
                a->data, a->shape->sizes[1],
                b->data, b->shape->sizes[1],
                0.0f,
                c->data, c->shape->sizes[1]);
    
    return c;
}
