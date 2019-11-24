/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leloss.h"
#include <assert.h>
#include <math.h>
#include "letensor-imp.h"

float
le_cross_entropy(LeTensor *h, LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(h->shape->sizes[1] == y->shape->sizes[1]);
    
    float result = 0.0f;
    unsigned i;
    
    unsigned elements_count = le_shape_get_elements_count(h->shape);
    for (i = 0; i < elements_count; i++)
    {
        float yi = le_tensor_f32_at(y, i);
        float hi = le_tensor_f32_at(h, i);
        result -= yi * log(hi) + (1.0f - yi) * log(1.0f - hi);
    }
    
    return result / elements_count;
}
