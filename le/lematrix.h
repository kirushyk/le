/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEMATRIX_H__
#define __LEMATRIX_H__

typedef struct LeMatrix LeMatrix;

LeMatrix * le_matrix_new                  (void);

LeMatrix * le_matrix_new_copy             (LeMatrix     *another);

LeMatrix * le_matrix_new_from_data        (unsigned      height,
                                           unsigned      width,
                                           const float  *data);

unsigned   le_matrix_get_width            (LeMatrix     *matrix);

unsigned   le_matrix_get_height           (LeMatrix     *matrix);

float      le_matrix_at                   (LeMatrix     *matrix,
                                           unsigned      y,
                                           unsigned      x);

void       le_matrix_set_element          (LeMatrix     *matrix,
                                           unsigned      y,
                                           unsigned      x,
                                           float         value);

LeMatrix * le_matrix_get_column           (LeMatrix     *matrix,
                                           unsigned      x);

LeMatrix * le_matrix_new_identity         (unsigned      size);

LeMatrix * le_matrix_new_uninitialized    (unsigned      height,
                                           unsigned      width);

LeMatrix * le_matrix_new_zeros            (unsigned      height,
                                           unsigned      width);

LeMatrix * le_matrix_new_rand             (unsigned      height,
                                           unsigned      width);

LeMatrix * le_matrix_new_transpose        (LeMatrix     *a);

LeMatrix * le_matrix_new_product          (LeMatrix     *a,
                                           LeMatrix     *b);

void       le_matrix_subtract             (LeMatrix     *a,
                                           LeMatrix     *b);

void       le_matrix_multiply_by_scalar   (LeMatrix     *a,
                                           float         b);

void       le_matrix_add_scalar           (LeMatrix     *a,
                                           float         b);

float      le_matrix_sum                  (LeMatrix     *matrix);

void       le_matrix_apply_sigmoid        (LeMatrix     *matrix);

void       le_matrix_apply_greater_than   (LeMatrix     *matrix,
                                           float         scalar);

void       le_matrix_apply_svm_prediction (LeMatrix     *matrix);

void       le_matrix_free                 (LeMatrix     *matrix);

/** @note: Temporary */
#include <stdio.h>

void       le_matrix_print                (LeMatrix     *matrix,
                                           FILE         *stream);

LeMatrix * le_matrix_new_polynomia        (LeMatrix     *matrix);

/** @note: Inner product of two column vectors */
float      le_dot_product                 (LeMatrix     *a,
                                           LeMatrix     *b);

float      le_rbf                         (LeMatrix     *a,
                                           LeMatrix     *b,
                                           float         sigma);

#endif
