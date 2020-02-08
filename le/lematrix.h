/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

/* Linear Algebra for two-dimensional Tensors */

#include "letensor.h"

#ifndef __LEMATRIX_H__
#define __LEMATRIX_H__

unsigned   le_matrix_get_width            (LeTensor     *matrix);

unsigned   le_matrix_get_height           (LeTensor     *matrix);

float      le_matrix_at                   (LeTensor     *matrix,
                                           unsigned      y,
                                           unsigned      x);

/// @note: Unlike le_tensor_add, supports horizontal broadcasting
void       le_matrix_add                  (LeTensor     *matrix,
                                           LeTensor     *another);

void       le_matrix_set_element          (LeTensor     *matrix,
                                           unsigned      y,
                                           unsigned      x,
                                           float         value);

LeTensor * le_matrix_new_from_data        (unsigned      height,
                                           unsigned      width,
                                           const float  *data);

LeTensor * le_matrix_new_identity         (unsigned      size);

LeTensor * le_matrix_new_uninitialized    (unsigned      height,
                                           unsigned      width);

LeTensor * le_matrix_new_zeros            (unsigned      height,
                                           unsigned      width);

LeTensor * le_matrix_new_rand             (unsigned      height,
                                           unsigned      width);

LeTensor * le_matrix_new_transpose        (LeTensor     *a);

LeTensor * le_matrix_new_sum              (LeTensor     *a,
                                           unsigned      dimension);

LeTensor * le_matrix_new_one_hot          (LeTensor     *a,
                                           unsigned      num_classes);

LeTensor * le_matrix_new_product          (LeTensor     *a,
                                           LeTensor     *b);

LeTensor * le_matrix_new_product_full     (LeTensor     *a,
                                           bool          transpose_a,
                                           LeTensor     *b,
                                           bool          transpose_b);

                                            
LeTensor * le_matrix_new_conv2d           (LeTensor     *image,
                                           LeTensor     *filter);

LeTensor * le_matrix_get_column           (LeTensor     *matrix,
                                           unsigned      x);

LeTensor * le_matrix_get_column_copy      (LeTensor     *matrix,
                                           unsigned      x);

void       le_matrix_apply_softmax        (LeTensor     *self);

#endif
