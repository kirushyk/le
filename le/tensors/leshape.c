/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leshape.h"
#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LeShape *
le_shape_new (unsigned num_dimensions, ...)
{
  LeShape *self = le_shape_new_uninitialized (num_dimensions);

  va_list args;
  va_start (args, num_dimensions);

  for (unsigned i = 0; i < num_dimensions; i++) {
    int size = va_arg (args, int);
    self->sizes[i] = size;
  }

  va_end (args);

  return self;
}

LeShape *
le_shape_new_uninitialized (unsigned num_dimensions)
{
  LeShape *self = g_new0 (LeShape, 1);
  self->num_dimensions = num_dimensions;
  self->sizes = g_new0 (gsize, num_dimensions);

  return self;
}

gsize *
le_shape_get_data (LeShape *shape)
{
  return shape->sizes;
}

gsize
le_shape_get_size (LeShape *shape, int dimension)
{
  g_assert_nonnull (shape);
  g_assert_cmpint (dimension, <, (int)shape->num_dimensions);
  g_assert_cmpint (dimension, >=, -(int)shape->num_dimensions);
  return shape->sizes[(dimension < 0) ? (shape->num_dimensions + dimension) : dimension];
}

void
le_shape_set_size (LeShape *shape, unsigned dimension, gsize size)
{
  assert (shape);
  assert (dimension < shape->num_dimensions);
  shape->sizes[dimension] = size;
}

LeShape *
le_shape_copy (LeShape *another)
{
  g_assert_nonnull (another);
  LeShape *self = g_new (LeShape, 1);
  self->num_dimensions = another->num_dimensions;
  if (self->num_dimensions > 0) {
    gsize size = self->num_dimensions * sizeof (gsize);
    self->sizes = g_malloc (size);
    memcpy (self->sizes, another->sizes, size);
  } else {
    self->sizes = NULL;
  }
  return self;
}

LeShape *
le_shape_lower_dimension (LeShape *another)
{
  /// @todo: Add assertions
  LeShape *self = g_new0 (LeShape, 1);
  self->num_dimensions = another->num_dimensions - 1;
  gsize size = self->num_dimensions * sizeof (gsize);
  self->sizes = g_malloc (size);
  memcpy (self->sizes, another->sizes + 1, size);
  return self;
}

void
le_shape_free (LeShape *self)
{
  if (self) {
    g_assert_cmpint (self->num_dimensions > 0, ==, self->sizes != NULL);

    if (self->sizes)
      g_free (self->sizes);
    g_free (self);
  }
}

const char *
le_shape_to_cstr (LeShape *shape)
{
  static char buffer[1024];

  if (shape) {
    char *ptr = buffer;
    ptr[0] = '(';
    /// @todo: Add overflow check
    ptr++;
    for (unsigned i = 0; i < shape->num_dimensions; i++) {
      int written = 0;
      if (shape->sizes[i]) {
        sprintf (ptr, "%d%n", shape->sizes[i], &written);
      } else {
        sprintf (ptr, "?");
        written = 1;
      }
      ptr += written;

      sprintf (ptr, "%s%n", i == (shape->num_dimensions - 1) ? ")" : ", ", &written);
      ptr += written;
    }
  } else {
    sprintf (buffer, "(null)");
  }

  return buffer;
}

gsize
le_shape_get_elements_count (LeShape *shape)
{
  gsize count = 0;
  if (shape) {
    g_assert_cmpint (shape->num_dimensions > 0, ==, shape->sizes != NULL);
    count = 1;
    for (unsigned i = 0; i < shape->num_dimensions; i++) {
      count *= shape->sizes[i];
    }
  }
  return count;
}

gsize
le_shape_get_regions_count (LeShape *shape)
{
  gsize count = 0;
  if (shape) {
    g_assert_cmpint (shape->num_dimensions > 0, ==, shape->sizes != NULL);
    count = 1;
    for (unsigned i = 0; i < shape->num_dimensions - 1; i++) {
      count *= shape->sizes[i];
    }
  }
  return count;
}

gboolean
le_shape_equal (LeShape *a, LeShape *b)
{
  g_assert_nonnull (a);
  g_assert_nonnull (b);

  if (a->num_dimensions != b->num_dimensions)
    return FALSE;

  for (unsigned i = 0; i < a->num_dimensions; i++) {
    if (a->sizes[i] != b->sizes[i]) {
      return FALSE;
    }
  }

  return TRUE;
}
