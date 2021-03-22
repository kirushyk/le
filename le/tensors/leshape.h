/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LESHAPE_H__
#define __LESHAPE_H__

#include <stdint.h>
#include <stdbool.h>
#include <le/lemacros.h>

LE_BEGIN_DECLS

typedef struct LeShape
{
    unsigned  num_dimensions;
    uint32_t *sizes;
} LeShape;

LeShape *    le_shape_new_uninitialized  (unsigned  num_dimensions);

LeShape *    le_shape_new                (unsigned  num_dimensions,
                                          ...);

uint32_t *   le_shape_get_data           (LeShape  *shape);

uint32_t     le_shape_get_size           (LeShape  *shape,
                                          unsigned  dimension);

void         le_shape_set_size           (LeShape  *shape,
                                          unsigned  dimension,
                                          uint32_t  size);

LeShape *    le_shape_copy               (LeShape  *shape);

LeShape *    le_shape_lower_dimension    (LeShape  *shape);

/// @note: Retrieves size in lowest order dimension
uint32_t     le_shape_get_last_size      (LeShape  *shape);

void         le_shape_free               (LeShape  *shape);

const char * le_shape_to_cstr            (LeShape  *shape);

uint32_t     le_shape_get_elements_count (LeShape  *shape);

/// @todo: Come up with a better name
uint32_t     le_shape_get_regions_count  (LeShape  *shape);

bool         le_shape_equal              (LeShape  *a,
                                          LeShape  *b);

LE_END_DECLS

#endif
