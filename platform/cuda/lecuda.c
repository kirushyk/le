/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lecuda.h"
#include <assert.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>
#include <cuda_runtime.h>

LeTensor *
le_cuda_matrix_new_product(LeTensor *a, LeTensor *b)
{
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    assert(a->shape->sizes[1] == b->shape->sizes[0]);
    
    LeTensor *c = le_matrix_new_uninitialized(LE_TYPE_FLOAT32, a->shape->sizes[0], b->shape->sizes[1]);
    
    return c;
}
