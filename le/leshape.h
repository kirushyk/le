/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdint.h>

#ifndef __LESHAPE_H__
#define __LESHAPE_H__

typedef struct LeShape
{
    unsigned  num_dimensions;
    uint32_t *sizes;
} LeShape;

LeShape * le_shape_new  (unsigned  num_dimensions,
                         ...);

LeShape * le_shape_copy (LeShape  *shape);

void      le_shape_free (LeShape  *shape);

#endif
