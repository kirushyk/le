/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdarg.h>
#include <stdbool.h>
#include <glib.h>
#include <glib-object.h>
#include "letype.h"
#include "leshape.h"

#ifndef __LETENSOR_H__
#define __LETENSOR_H__

G_BEGIN_DECLS

/// Tensors are multi-dimensional arrays containing elements of a uniform type
G_DECLARE_FINAL_TYPE (LeTensor, le_tensor, LE, TENSOR, GObject);

/// @note: Make sure to pass correct number of parameters
LeTensor *         le_tensor_new                 (LeType           element_type,
                                                  gsize            num_dimensions,
                                                  ...);

LeTensor *         le_tensor_new_from_va_list    (LeType           element_type,
                                                  gsize            num_dimensions,
                                                  va_list          dims_and_data);

/// @note: Takes ownership of shape
LeTensor *         le_tensor_new_uninitialized   (LeType           element_type,
                                                  LeShape *        shape);

void *             le_tensor_get_data            (const LeTensor * another);

LeTensor *         le_tensor_new_copy            (const LeTensor * another);

LeTensor *         le_tensor_new_zeros           (LeType           element_type,
                                                  LeShape *        shape);

LeTensor *         le_tensor_new_zeros_like      (const LeTensor * another);

/// @note: Takes ownership of shape
LeTensor *         le_tensor_new_rand_f32        (LeShape *        shape);

LeTensor *         le_tensor_new_cast            (LeTensor *       tensor,
                                                  LeType           type);

bool               le_tensor_contiguous          (const LeTensor * tensor);

bool               le_tensor_equal               (const LeTensor * a,
                                                  const LeTensor * b);

bool               le_tensor_reshape             (LeTensor *       tensor,
                                                  gsize            num_dimensions,
                                                  ...);

LeTensor *         le_tensor_pick                (LeTensor *       another,
                                                  guint32          index);

LeTensor *         le_tensor_pick_copy           (const LeTensor * another,
                                                  guint32          index);

void *             le_tensor_at                  (const LeTensor * another,
                                                  guint32          index);

guint8            le_tensor_at_u8                (const LeTensor * tensor,
                                                  guint32          index);

guint32           le_tensor_at_u32               (const LeTensor * tensor,
                                                  guint32          index);

gfloat              le_tensor_at_f32             (const LeTensor * tensor,
                                                  guint32          index);

void               le_tensor_assign              (LeTensor *       tensor,
                                                  const LeTensor * another);

void               le_tensor_set_f32             (LeTensor *       tensor,
                                                  guint32          index,
                                                  gfloat           value);

/// @note: a = a + b
void               le_tensor_add_f32             (LeTensor *       a,
                                                  gfloat           b);

/// @note: a = a + b
void               le_tensor_add_tensor          (LeTensor *       a,
                                                  const LeTensor * b);

/// @note: a = a + b
#define le_tensor_add(a, b) _Generic(b, \
   gfloat: le_tensor_add_f32, \
   LeTensor *: le_tensor_add_tensor, \
   const LeTensor *: le_tensor_add_tensor \
)(a, b)

/// @note: a = a - b
void               le_tensor_sub_f32             (LeTensor *       a,
                                                  gfloat           b);

/// @note: a = a - b
void               le_tensor_sub_tensor          (LeTensor *       a,
                                                  const LeTensor * b);

/// @note: a = a - b
#define le_tensor_sub(a, b) _Generic(b, \
   gfloat: le_tensor_sub_f32, \
   LeTensor *: le_tensor_sub_tensor, \
   const LeTensor *: le_tensor_sub_tensor \
)(a, b)

/// @note: a = a - c * b
void               le_tensor_sub_scaled_f32      (LeTensor *       a,
                                                  gfloat           c,
                                                  const LeTensor * b);

#define le_tensor_sub_scaled(a, s, b) \
    _Generic(s, \
        gfloat: le_tensor_sub_scaled_f32 \
    )(a, s, b)
    
/// @note: a = a * b
void               le_tensor_mul_f32             (LeTensor *       a,
                                                  gfloat           b);

/// @note: a = a * b
void               le_tensor_mul_tensor          (LeTensor *       a,
                                                  const LeTensor * b);

/// @note: a = a * b
#define le_tensor_mul(a, b) _Generic(b, \
   gfloat: le_tensor_mul_f32, \
   LeTensor *: le_tensor_mul_tensor, \
   const LeTensor *: le_tensor_mul_tensor \
)(a, b)

void               le_tensor_div_u32             (LeTensor *       a,
                                                  guint32          b);

gfloat              le_tensor_sum_f32            (const LeTensor * tensor);

gfloat              le_tensor_sad_f32            (const LeTensor * a,
                                                  const LeTensor * b);

gfloat              le_tensor_l2_f32             (const LeTensor * tensor);

void               le_tensor_apply_sigmoid       (LeTensor *       tensor);

void               le_tensor_apply_sigmoid_prime (LeTensor *       tensor);

void               le_tensor_apply_tanh          (LeTensor *       tensor);

void               le_tensor_apply_sqr           (LeTensor *       tensor);

void               le_tensor_apply_1_minus       (LeTensor *       tensor);

void               le_tensor_apply_x_minus_sqr_x (LeTensor *       tensor);

void               le_tensor_apply_gt_f32        (LeTensor *       tensor,
                                                  gfloat           scalar);

/// @note: a = a > b
#define le_tensor_apply_gt(tensor, scalar) _Generic(scalar, \
   gfloat: le_tensor_apply_gt_f32 \
)(tensor, scalar)

void               le_tensor_apply_sgn           (LeTensor *       tensor);

void               le_tensor_apply_relu          (LeTensor *       tensor);

LeTensor *         le_tensor_ref                 (LeTensor *       tensor);

void               le_tensor_unref               (LeTensor *       tensor);

/** @section ugly */
#include <stdio.h>

const char *       le_tensor_to_cstr             (const LeTensor * tensor);

void               le_tensor_print               (const LeTensor * tensor,
                                                  FILE *           stream);

/** @note: Inner product of two column vectors */
gfloat              le_dot_product               (const LeTensor * a,
                                                  const LeTensor * b);

gfloat              le_rbf                       (const LeTensor * a,
                                                  const LeTensor * b,
                                                  gfloat           sigma);

typedef struct LeTensorStats
{
   gfloat min;
   gfloat max;
   gfloat mean;
   gfloat deviation;
   gsize nans;
   gsize zeros;
} LeTensorStats;

LeTensorStats      le_tensor_get_stats           (LeTensor *       tensor);

LeTensor *         le_tensor_new_equal_u8        (LeType           type,
                                                  LeTensor *       tensor,
                                                  guint8           scalar);

G_END_DECLS

#endif
