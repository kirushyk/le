/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lematrix.h"
#include <assert.h>
#include "letensor-imp.h"

float
le_matrix_at(LeTensor *self, unsigned y, unsigned x)
{
    assert(self->shape->num_dimensions == 2);
    
    assert(y < self->shape->sizes[0]);
    assert(x < self->shape->sizes[1]);
    
    return self->data[y * self->shape->sizes[1] + x];
}
