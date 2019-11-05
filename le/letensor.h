/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "letype.h"

#ifndef __LEMATRIX_H__
#define __LEMATRIX_H__

typedef struct LeTensor LeTensor;

LeTensor * le_matrix_new                  (void);

LeTensor * le_matrix_new_copy             (LeTensor     *another);

LeTensor * le_matrix_new_from_data        (unsigned      height,
                                           unsigned      width,
                                           const float  *data);

unsigned   le_matrix_get_width            (LeTensor     *matrix);

unsigned   le_matrix_get_height           (LeTensor     *matrix);

float      le_matrix_at                   (LeTensor     *matrix,
                                           unsigned      y,
                                           unsigned      x);

void       le_matrix_set_element          (LeTensor     *matrix,
                                           unsigned      y,
                                           unsigned      x,
                                           float         value);

LeTensor * le_matrix_get_column           (LeTensor     *matrix,
                                           unsigned      x);

LeTensor * le_matrix_new_identity         (unsigned      size);

LeTensor * le_matrix_new_uninitialized    (unsigned      height,
                                           unsigned      width);

LeTensor * le_matrix_new_zeros            (unsigned      height,
                                           unsigned      width);

LeTensor * le_matrix_new_rand             (unsigned      height,
                                           unsigned      width);

LeTensor * le_matrix_new_transpose        (LeTensor     *a);

LeTensor * le_matrix_new_product          (LeTensor     *a,
                                           LeTensor     *b);

void       le_matrix_subtract             (LeTensor     *a,
                                           LeTensor     *b);

void       le_matrix_multiply_by_scalar   (LeTensor     *a,
                                           float         b);

void       le_matrix_add_scalar           (LeTensor     *a,
                                           float         b);

float      le_matrix_sum                  (LeTensor     *matrix);

void       le_matrix_apply_sigmoid        (LeTensor     *matrix);

void       le_matrix_apply_greater_than   (LeTensor     *matrix,
                                           float         scalar);

void       le_matrix_apply_svm_prediction (LeTensor     *matrix);

void       le_matrix_free                 (LeTensor     *matrix);

/** @note: Temporary */
#include <stdio.h>

void       le_matrix_print                (LeTensor     *matrix,
                                           FILE         *stream);

LeTensor * le_matrix_new_polynomia        (LeTensor     *matrix);

/** @note: Inner product of two column vectors */
float      le_dot_product                 (LeTensor     *a,
                                           LeTensor     *b);

float      le_rbf                         (LeTensor     *a,
                                           LeTensor     *b,
                                           float         sigma);

#endif
