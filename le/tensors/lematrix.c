/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lematrix.h"
#include "../config.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <le/math/lerand.h>
#include "letensor-imp.h"
#ifdef __APPLE__
#  include "../backends/accelerate/leaccelerate.h"
#elif defined(HAVE_OPENBLAS)
#  include "../backends/openblas/leopenblas.h"
#endif
#if defined(HAVE_CUDA)
#  include "../backends/cuda/lecuda.h"
#endif
#if defined(HAVE_METAL)
#  include "../backends/metal/lemetal.h"
#endif

unsigned
le_matrix_get_width (const LeTensor *self)
{
  assert (self->shape->num_dimensions == 2);

  return self->shape->sizes[1];
}

unsigned
le_matrix_get_height (const LeTensor *self)
{
  assert (self->shape->num_dimensions == 2);

  return self->shape->sizes[0];
}

gfloat
le_matrix_at_f32 (const LeTensor *self, unsigned y, unsigned x)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_F32);
  assert (self->shape->num_dimensions == 2);
  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);

  return ((gfloat *)self->data)[y * self->stride + x];
}

gdouble
le_matrix_at_f64 (const LeTensor *self, unsigned y, unsigned x)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_F64);
  assert (self->shape->num_dimensions == 2);
  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);

  return ((gdouble *)self->data)[y * self->stride + x];
}

gint8
le_matrix_at_i8 (const LeTensor *self, unsigned y, unsigned x)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_I8);
  assert (self->shape->num_dimensions == 2);
  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);

  return ((gint8 *)self->data)[y * self->stride + x];
}

gint16
le_matrix_at_i16 (const LeTensor *self, unsigned y, unsigned x)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_I16);
  assert (self->shape->num_dimensions == 2);
  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);

  return ((gint16 *)self->data)[y * self->stride + x];
}

gint32
le_matrix_at_i32 (const LeTensor *self, unsigned y, unsigned x)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_I32);
  assert (self->shape->num_dimensions == 2);
  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);

  return ((gint32 *)self->data)[y * self->stride + x];
}

guint32
le_matrix_at_u32 (const LeTensor *self, unsigned y, unsigned x)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_U32);
  assert (self->shape->num_dimensions == 2);
  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);

  return ((guint32 *)self->data)[y * self->stride + x];
}

void
le_matrix_add (LeTensor *self, const LeTensor *another)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (another->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_F32);
  assert (self->stride == le_shape_get_size (self->shape, -1));
  assert (another->stride == le_shape_get_size (another->shape, -1));
  assert (self->shape->num_dimensions == 2);

  /// @note: Add horizontal broadcasting
  assert (self->shape->sizes[0] == another->shape->sizes[0]);

  if (le_shape_equal (self->shape, another->shape)) {
    le_tensor_add (self, another);
  } else if (another->shape->sizes[1] == 1) {
    for (guint32 y = 0; y < self->shape->sizes[0]; y++) {
      for (guint32 x = 0; x < self->shape->sizes[1]; x++) {
        ((gfloat *)self->data)[y * self->stride + x] += ((gfloat *)another->data)[y * another->stride];
      }
    }
  } else {
    assert (false);
  }
}

void
le_matrix_set_i8 (LeTensor *self, unsigned y, unsigned x, gint8 value)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_I8);
  assert (self->shape->num_dimensions == 2);

  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);
  assert (x < self->stride);

  /// @todo: Take stride into account
  ((gint8 *)self->data)[y * self->stride + x] = value;
}

void
le_matrix_set_u8 (LeTensor *self, unsigned y, unsigned x, guint8 value)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_U8);
  assert (self->shape->num_dimensions == 2);

  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);
  assert (x < self->stride);

  /// @todo: Take stride into account
  ((guint8 *)self->data)[y * self->stride + x] = value;
}

void
le_matrix_set_i16 (LeTensor *self, unsigned y, unsigned x, gint16 value)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_I16);
  assert (self->shape->num_dimensions == 2);

  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);
  assert (x < self->stride);

  /// @todo: Take stride into account
  ((gint16 *)self->data)[y * self->stride + x] = value;
}

void
le_matrix_set_u16 (LeTensor *self, unsigned y, unsigned x, guint16 value)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_U16);
  assert (self->shape->num_dimensions == 2);

  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);
  assert (x < self->stride);

  /// @todo: Take stride into account
  ((guint16 *)self->data)[y * self->stride + x] = value;
}

void
le_matrix_set_i32 (LeTensor *self, unsigned y, unsigned x, gint32 value)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_I32);
  assert (self->shape->num_dimensions == 2);

  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);
  assert (x < self->stride);

  /// @todo: Take stride into account
  ((gint32 *)self->data)[y * self->stride + x] = value;
}

void
le_matrix_set_u32 (LeTensor *self, unsigned y, unsigned x, guint32 value)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_U32);
  assert (self->shape->num_dimensions == 2);

  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);
  assert (x < self->stride);

  /// @todo: Take stride into account
  ((guint32 *)self->data)[y * self->stride + x] = value;
}

void
le_matrix_set_f16 (LeTensor *self, unsigned y, unsigned x, lehalf value)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_F16);
  assert (self->shape->num_dimensions == 2);

  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);
  assert (x < self->stride);

  /// @todo: Take stride into account
  ((lehalf *)self->data)[y * self->stride + x] = value;
}

void
le_matrix_set_f32 (LeTensor *self, unsigned y, unsigned x, gfloat value)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_F32);
  assert (self->shape->num_dimensions == 2);

  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);
  assert (x < self->stride);

  /// @todo: Take stride into account
  ((gfloat *)self->data)[y * self->stride + x] = value;
}

void
le_matrix_set_f64 (LeTensor *self, unsigned y, unsigned x, gdouble value)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->element_type == LE_TYPE_F64);
  assert (self->shape->num_dimensions == 2);

  assert (y < self->shape->sizes[0]);
  assert (x < self->shape->sizes[1]);
  assert (x < self->stride);

  /// @todo: Take stride into account
  ((gdouble *)self->data)[y * self->stride + x] = value;
}

LeTensor *
le_matrix_new_identity (LeType type, unsigned size)
{
  unsigned x;
  unsigned y;
  LeTensor *self;

  self = g_object_new (le_tensor_get_type (), NULL);
  self->device_type = LE_DEVICE_TYPE_CPU;
  self->element_type = LE_TYPE_F32;
  self->shape = le_shape_new (2, size, size);
  self->stride = size;
  self->owns_data = true;
  self->data = g_new0 (gfloat, size * size);

  for (y = 0; y < size; y++) {
    for (x = 0; x < size; x++) {
      switch (type) {
      case LE_TYPE_I8:
        ((gint8 *)self->data)[y * size + x] = (x == y) ? 1 : 0;
        break;
      case LE_TYPE_U8:
        ((guint8 *)self->data)[y * size + x] = (x == y) ? 1 : 0;
        break;
      case LE_TYPE_I16:
        ((gint16 *)self->data)[y * size + x] = (x == y) ? 1 : 0;
        break;
      case LE_TYPE_U16:
        ((guint16 *)self->data)[y * size + x] = (x == y) ? 1 : 0;
        break;
      case LE_TYPE_I32:
        ((gint32 *)self->data)[y * size + x] = (x == y) ? 1 : 0;
        break;
      case LE_TYPE_U32:
        ((guint32 *)self->data)[y * size + x] = (x == y) ? 1 : 0;
        break;
      case LE_TYPE_F16:
        ((guint16 *)self->data)[y * size + x] = (x == y) ? F16_1 : F16_0;
        break;
      case LE_TYPE_F32:
        ((gfloat *)self->data)[y * size + x] = (x == y) ? 1.0f : 0.0f;
        break;
      case LE_TYPE_F64:
        ((gdouble *)self->data)[y * size + x] = (x == y) ? 1.0 : 0.0;
        break;
      case LE_TYPE_VOID:
      default:
        break;
      }
    }
  }

  return self;
}

LeTensor *
le_matrix_new_uninitialized (LeType type, unsigned height, unsigned width)
{
  LeTensor *self;

  self = g_object_new (le_tensor_get_type (), NULL);
  self->device_type = LE_DEVICE_TYPE_CPU;
  self->element_type = type;
  self->shape = le_shape_new (2, height, width);
  self->stride = width;
  self->owns_data = true;
  self->data = g_malloc (height * width * le_type_size (self->element_type));

  return self;
}

LeTensor *
le_matrix_new_zeros (LeType type, unsigned height, unsigned width)
{
  unsigned i;
  unsigned elements_count;
  LeTensor *self;

  self = g_object_new (le_tensor_get_type (), NULL);
  self->device_type = LE_DEVICE_TYPE_CPU;
  self->shape = le_shape_new (2, height, width);
  self->element_type = LE_TYPE_F32;
  self->stride = width;
  self->owns_data = true;
  self->data = g_new0 (gfloat, height * width);
  elements_count = height * width;

  for (i = 0; i < elements_count; i++) {
    switch (type) {
    case LE_TYPE_I8:
      ((gint8 *)self->data)[i] = 0;
      break;
    case LE_TYPE_U8:
      ((guint8 *)self->data)[i] = 0;
      break;
    case LE_TYPE_I16:
      ((gint16 *)self->data)[i] = 0;
      break;
    case LE_TYPE_U16:
      ((guint16 *)self->data)[i] = 0;
      break;
    case LE_TYPE_I32:
      ((gint32 *)self->data)[i] = 0;
      break;
    case LE_TYPE_U32:
      ((guint32 *)self->data)[i] = 0;
      break;
    case LE_TYPE_F16:
      ((guint16 *)self->data)[i] = F16_0;
      break;
    case LE_TYPE_F32:
      ((gfloat *)self->data)[i] = 0.0f;
      break;
    case LE_TYPE_F64:
      ((gdouble *)self->data)[i] = 0.0;
      break;
    case LE_TYPE_VOID:
    default:
      break;
    }
  }

  return self;
}

LeTensor *
le_matrix_new_rand_f32 (LeDistribution distribution, unsigned height, unsigned width)
{
  unsigned i;
  unsigned elements_count;
  LeTensor *self;

  self = g_object_new (le_tensor_get_type (), NULL);
  self->device_type = LE_DEVICE_TYPE_CPU;
  self->element_type = LE_TYPE_F32;
  self->shape = le_shape_new (2, height, width);
  self->stride = width;
  self->owns_data = true;
  elements_count = height * width;
  self->data = g_new0 (gfloat, elements_count);

  for (i = 0; i < elements_count; i++) {
    ((gfloat *)self->data)[i] = le_random_f32 (distribution);
  }

  return self;
}

#define TRNASPOSE(type)                                                                                                \
  for (y = 0; y < self->shape->sizes[0]; y++) {                                                                        \
    for (x = 0; x < self->shape->sizes[1]; x++) {                                                                      \
      ((type *)self->data)[y * self->shape->sizes[1] + x] = ((type *)a->data)[x * a->stride + y];                      \
    }                                                                                                                  \
  }

LeTensor *
le_matrix_new_transpose (LeTensor *a)
{
  assert (a->device_type == LE_DEVICE_TYPE_CPU);
  assert (a->shape->num_dimensions == 2);

  unsigned x;
  unsigned y;
  LeTensor *self;

  self = g_object_new (le_tensor_get_type (), NULL);
  self->device_type = LE_DEVICE_TYPE_CPU;
  self->element_type = a->element_type;
  self->shape = le_shape_new (2, a->shape->sizes[1], a->shape->sizes[0]);
  self->stride = le_shape_get_size (self->shape, -1);
  self->owns_data = true;
  self->data = g_malloc (le_shape_get_elements_count (self->shape) * le_type_size (self->element_type));

  switch (le_type_size (self->element_type)) {
  case 1:
    TRNASPOSE (guint8);
    break;

  case 2:
    TRNASPOSE (guint16);
    break;

  case 4:
    TRNASPOSE (guint32);
    break;

  case 8:
    TRNASPOSE (guint64);
    break;

  default:
    break;
  }

  return self;
}

#undef TRANSPOSE

LeTensor *
le_matrix_new_sum (const LeTensor *a, unsigned dimension)
{
  /// @todo: Take stride into account
  assert (a->device_type == LE_DEVICE_TYPE_CPU);

  assert (a->shape->num_dimensions == 2);

  LeTensor *self = g_object_new (le_tensor_get_type (), NULL);
  self->device_type = LE_DEVICE_TYPE_CPU;
  self->element_type = LE_TYPE_F32;
  self->shape = le_shape_new (2, a->shape->sizes[0], 1 /*a->shape->sizes[1]*/);
  self->stride = le_shape_get_size (self->shape, -1);
  self->owns_data = true;
  self->data = g_malloc (le_shape_get_elements_count (self->shape) * le_type_size (self->element_type));

  assert (/*(dimension == 0) || */ (dimension == 1));
  for (unsigned y = 0; y < a->shape->sizes[0]; y++) {
    ((gfloat *)self->data)[y] = 0.0f;
    for (unsigned x = 0; x < a->shape->sizes[1]; x++) {
      ((gfloat *)self->data)[y] += ((gfloat *)a->data)[y * a->shape->sizes[1] + x];
    }
  }

  return self;
}

LeTensor *
le_matrix_new_one_hot (LeType type, const LeTensor *a, unsigned num_classes)
{
  /// @todo: Take stride into account
  assert (a->device_type == LE_DEVICE_TYPE_CPU);
  assert (a->shape->num_dimensions == 2);
  assert (a->shape->sizes[0] == 1);

  unsigned example, klass;
  LeTensor *self;

  self = g_object_new (le_tensor_get_type (), NULL);
  self->device_type = LE_DEVICE_TYPE_CPU;
  self->element_type = type;
  self->shape = le_shape_new (2, num_classes, a->shape->sizes[1]);
  self->stride = le_shape_get_size (self->shape, -1);
  self->owns_data = true;
  self->data = g_malloc (le_shape_get_elements_count (self->shape) * le_type_size (self->element_type));

  for (example = 0; example < a->shape->sizes[1]; example++) {
    for (klass = 0; klass < num_classes; klass++) {
      bool hot = (klass == le_tensor_at_u8 (a, example));
      switch (type) {
      case LE_TYPE_I8:
        le_matrix_set_i8 (self, klass, example, hot ? 1 : 0);
        break;
      case LE_TYPE_U8:
        le_matrix_set_u8 (self, klass, example, hot ? 1 : 0);
        break;
      case LE_TYPE_I16:
        le_matrix_set_i16 (self, klass, example, hot ? 1 : 0);
        break;
      case LE_TYPE_U16:
        le_matrix_set_u16 (self, klass, example, hot ? 1 : 0);
        break;
      case LE_TYPE_I32:
        le_matrix_set_i32 (self, klass, example, hot ? 1 : 0);
        break;
      case LE_TYPE_U32:
        le_matrix_set_u32 (self, klass, example, hot ? 1 : 0);
        break;
      case LE_TYPE_F16:
        le_matrix_set_f16 (self, klass, example, hot ? F16_1 : F16_0);
        break;
      case LE_TYPE_F32:
        le_matrix_set_f32 (self, klass, example, hot ? 1.0f : 0.0f);
        break;
      case LE_TYPE_F64:
        le_matrix_set_f32 (self, klass, example, hot ? 1.0 : 0.0);
        break;
      case LE_TYPE_VOID:
      default:
        break;
      }
    }
  }

  return self;
}

LeTensor *
le_matrix_new_product (const LeTensor *a, const LeTensor *b)
{
  return le_matrix_new_product_full (a, false, b, false);
}

LeTensor *
le_matrix_new_product_full (const LeTensor *a, bool transpose_a, const LeTensor *b, bool transpose_b)
{
  assert (a->device_type == b->device_type);

  switch (a->device_type) {
#ifdef HAVE_METAL
  case LE_DEVICE_TYPE_METAL:
    return le_metal_matrix_new_product (a, transpose_a, b, transpose_b);
#endif

#if defined(HAVE_CUDA)
  case LE_DEVICE_TYPE_CUDA:
    return le_cuda_matrix_new_product (a, transpose_a, b, transpose_b);
#endif

  case LE_DEVICE_TYPE_CPU:
    /// @todo: Take stride into account
#ifdef __APPLE__
    return le_accelerate_matrix_new_product (a, transpose_a, b, transpose_b);
#elif defined(HAVE_OPENBLAS)
    return le_openblas_matrix_new_product (a, transpose_a, b, transpose_b);
#else
    assert (a->element_type == b->element_type);
    assert (a->shape->num_dimensions == 2);
    assert (b->shape->num_dimensions == 2);
    {
      unsigned a_width = transpose_a ? a->shape->sizes[0] : a->shape->sizes[1];
      unsigned a_height = transpose_a ? a->shape->sizes[1] : a->shape->sizes[0];
      unsigned b_width = transpose_b ? b->shape->sizes[0] : b->shape->sizes[1];
      unsigned b_height = transpose_b ? b->shape->sizes[1] : b->shape->sizes[0];

      assert (a_width == b_height);

      LeTensor *self = g_object_new (le_tensor_get_type (), NULL);
      self->device_type = LE_DEVICE_TYPE_CPU;
      self->element_type = a->element_type;
      self->shape = le_shape_new (2, a_height, b_width);
      self->stride = le_shape_get_size (self->shape, -1);
      self->owns_data = true;
      self->data = g_new0 (gfloat, le_shape_get_elements_count (self->shape));

      for (unsigned y = 0; y < a_height; y++) {
        for (unsigned x = 0; x < b_width; x++) {
          gsize self_index = y * self->stride + x;
          ((gfloat *)self->data)[self_index] = 0.0f;
          for (unsigned i = 0; i < a_width; i++) {
            /// @note: Check indices
            gsize a_index = transpose_a ? i * a->stride + y : y * a->stride + i;
            gfloat a_element = ((gfloat *)a->data)[a_index];
            gsize b_index = transpose_b ? x * b->stride + i : i * b->stride + x;
            gfloat b_element = ((gfloat *)b->data)[b_index];
            gfloat prod = a_element * b_element;
            ((gfloat *)self->data)[self_index] += prod;
          }
        }
      }

      return self;
    }
#endif

  default:
    assert (false);
  }
  assert (false);

  return NULL;
}

LeTensor *
le_matrix_new_conv2d (const LeTensor *image, const LeTensor *filter)
{
  assert (image->device_type == LE_DEVICE_TYPE_CPU);
  assert (filter->device_type == LE_DEVICE_TYPE_CPU);
  assert (image->shape->num_dimensions == 2);
  assert (filter->shape->num_dimensions == 2);
  assert (image->element_type == LE_TYPE_F32);
  assert (filter->element_type == LE_TYPE_F32);

  gint32 fh = le_matrix_get_height (filter);
  gint32 height = le_matrix_get_height (image) - fh + 1;
  gint32 fw = le_matrix_get_width (filter);
  gint32 width = le_matrix_get_width (image) - le_matrix_get_width (filter) + 1;

  assert (height > 1);
  assert (width > 1);

  LeTensor *self = g_object_new (le_tensor_get_type (), NULL);
  self->device_type = LE_DEVICE_TYPE_CPU;
  self->element_type = image->element_type;
  self->shape = le_shape_new (2, height, width);
  self->stride = le_shape_get_size (self->shape, -1);
  self->owns_data = true;
  self->data = g_new0 (gfloat, le_shape_get_elements_count (self->shape));

  for (gint32 oy = 0; oy < height; oy++) {
    for (gint32 ox = 0; ox < width; ox++) {
      gfloat value = 0.0f;
      for (gint32 fy = 0; fy < fh; fy++) {
        for (gint32 fx = 0; fx < fw; fx++) {
          value += le_matrix_at_f32 (image, oy + fy, ox + fx) * le_matrix_at_f32 (filter, fy, fx);
        }
      }
      le_matrix_set (self, oy, ox, value);
    }
  }

  return self;
}

LeTensor *
le_matrix_get_column (const LeTensor *matrix, unsigned x)
{
  /// @todo: Take stride into account
  assert (matrix->device_type == LE_DEVICE_TYPE_CPU);
  assert (matrix->shape->num_dimensions == 2);

  LeTensor *self = g_object_new (le_tensor_get_type (), NULL);
  self->device_type = LE_DEVICE_TYPE_CPU;
  self->element_type = matrix->element_type;
  self->shape = le_shape_new (2, matrix->shape->sizes[0], 1);
  self->stride = matrix->stride;

  self->owns_data = false;
  self->data = matrix->data + x * le_type_size (self->element_type);

  return self;
}

LeTensor *
le_matrix_get_column_copy (const LeTensor *self, unsigned x)
{
  /// @todo: Take stride into account
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->shape->num_dimensions == 2);
  assert (self->element_type == LE_TYPE_F32);
  /// @todo: Add dimension checks

  unsigned y;
  unsigned height = le_matrix_get_height (self);
  LeTensor *column = le_matrix_new_uninitialized (self->element_type, height, 1);

  for (y = 0; y < height; y++) {
    ((gfloat *)column->data)[y] = ((gfloat *)self->data)[y * self->stride + x];
  }

  return column;
}

LeTensor *
le_matrix_get_columns_copy (const LeTensor *self, unsigned x, unsigned width)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->shape->num_dimensions == 2);
  unsigned self_width = le_matrix_get_width (self);
  assert (self_width >= x + width);

  unsigned height = le_matrix_get_height (self);
  const gsize element_size = le_type_size (self->element_type);

  LeTensor *columns = le_matrix_new_uninitialized (self->element_type, height, width);
  for (unsigned y = 0; y < height; y++) {
    memcpy ((char *)columns->data + y * width * element_size,
        (char *)self->data + (x + y * self->stride) * element_size, width * element_size);
  }

  return columns;
}

void
le_matrix_apply_softmax (LeTensor *self)
{
  assert (self->device_type == LE_DEVICE_TYPE_CPU);
  assert (self->shape->num_dimensions == 2);

  /// @todo: Take stride into account
  unsigned example, klass;
  unsigned num_classes = self->shape->sizes[0];
  unsigned num_examples = self->shape->sizes[1];

  for (example = 0; example < num_examples; example++) {
    gfloat max = -INFINITY;
    for (klass = 0; klass < num_classes; klass++) {
      gfloat value = le_matrix_at_f32 (self, klass, example);
      if (value > max) {
        max = value;
      }
    }
    gfloat sum = 0;
    for (klass = 0; klass < num_classes; klass++) {
      gfloat activation = expf (le_matrix_at_f32 (self, klass, example) - max);
      sum += activation;
      le_matrix_set (self, klass, example, activation);
    }
    for (klass = 0; klass < num_classes; klass++) {
      gfloat activation = le_matrix_at_f32 (self, klass, example);
      activation /= sum;
      le_matrix_set (self, klass, example, activation);
    }
  }
}
