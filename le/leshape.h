/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdint.h>
#include <stdbool.h>

#ifndef __LESHAPE_H__
#define __LESHAPE_H__

typedef struct LeShape
{
    unsigned  num_dimensions;
    uint32_t *sizes;
} LeShape;

LeShape *    le_shape_new                (unsigned  num_dimensions,
                                          ...);

/// @note: Takes ownership of sizes array
LeShape *    le_shape_new_from_data      (unsigned  num_dimensions,
                                          uint32_t *sizes);

LeShape *    le_shape_copy               (LeShape  *shape);

LeShape *    le_shape_lower_dimension    (LeShape  *shape);

/// @note: Retrieves size in lowest order dimension
uint32_t     le_shape_get_last_size      (LeShape  *shape);

void         le_shape_free               (LeShape  *shape);

const char * le_shape_to_cstr            (LeShape  *shape);

uint32_t     le_shape_get_elements_count (LeShape  *shape);

bool         le_shape_equal              (LeShape  *a,
                                          LeShape  *b);

#endif
