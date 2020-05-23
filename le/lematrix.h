/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

/* Linear Algebra for two-dimensional Tensors */

#ifndef __LEMATRIX_H__
#define __LEMATRIX_H__

#include "lemacros.h"
#include "letensor.h"

LE_BEGIN_DECLS

unsigned   le_matrix_get_width            (const LeTensor *matrix);

unsigned   le_matrix_get_height           (const LeTensor *matrix);

float      le_matrix_at_f32                   (const LeTensor *matrix,
                                           unsigned      y,
                                           unsigned      x);

/// @note: Unlike le_tensor_add, supports horizontal broadcasting
void       le_matrix_add                  (LeTensor     *matrix,
                                           const LeTensor *another);

void       le_matrix_set_f32          (LeTensor     *matrix,
                                           unsigned      y,
                                           unsigned      x,
                                           float         value);

LeTensor * le_matrix_new_identity         (LeType type,
                                            unsigned      size);

LeTensor * le_matrix_new_uninitialized_f32    (unsigned      height,
                                           unsigned      width);

LeTensor * le_matrix_new_zeros_f32            (unsigned      height,
                                           unsigned      width);

LeTensor * le_matrix_new_rand_f32             (unsigned      height,
                                           unsigned      width);

LeTensor * le_matrix_new_transpose        (LeTensor     *a);

LeTensor * le_matrix_new_sum              (const LeTensor     *a,
                                           unsigned      dimension);

LeTensor * le_matrix_new_one_hot_u8f32          (const LeTensor     *a,
                                           unsigned      num_classes);

LeTensor * le_matrix_new_product          (const LeTensor     *a,
                                           const LeTensor     *b);

LeTensor * le_matrix_new_product_full     (const LeTensor     *a,
                                           bool                transpose_a,
                                           const LeTensor     *b,
                                           bool                transpose_b);

                                            
LeTensor * le_matrix_new_conv2d           (const LeTensor     *image,
                                           const LeTensor     *filter);

LeTensor * le_matrix_get_column           (const LeTensor *matrix,
                                           unsigned        x);

LeTensor * le_matrix_get_column_copy      (LeTensor     *matrix,
                                           unsigned      x);

void       le_matrix_apply_softmax        (LeTensor     *self);

void       le_matrix_apply_softmax_prime  (LeTensor     *matrix);

LE_END_DECLS

#endif
