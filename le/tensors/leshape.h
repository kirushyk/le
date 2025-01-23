/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_TENSORS_SHAPE_H__
#define __LE_TENSORS_SHAPE_H__

#include <stdint.h>
#include <glib.h>
#include <glib-object.h>

G_BEGIN_DECLS

G_DECLARE_FINAL_TYPE (LeShape, le_shape, LE, SHAPE, GObject);

struct _LeShape
{
  GObject  parent;
  gsize    num_dimensions;
  gsize   *sizes;
};

/// @todo: Rename, update tensorlist
LeShape *    le_shape_new_uninitialized  (gsize     num_dimensions);

LeShape *    le_shape_new                (gsize     num_dimensions,
                                          ...);

gsize *      le_shape_get_data           (LeShape  *shape);

gsize        le_shape_get_size           (LeShape  *shape,
                                          gint      dimension);

void         le_shape_set_size           (LeShape  *shape,
                                          gsize     dimension,
                                          gsize     size);

LeShape *    le_shape_copy               (LeShape  *shape);

LeShape *    le_shape_lower_dimension    (LeShape  *shape);

void         le_shape_unref              (LeShape  *shape);

const char * le_shape_to_cstr            (LeShape  *shape);

gsize        le_shape_get_elements_count (LeShape  *shape);

/// @todo: Come up with a better name
gsize        le_shape_get_regions_count  (LeShape  *shape);

gboolean     le_shape_equal              (LeShape  *a,
                                          LeShape  *b);

G_END_DECLS

#endif
