/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "letype.h"

#ifndef __LETENSOR_H__
#define __LETENSOR_H__

typedef struct LeTensor LeTensor;

LeTensor * le_matrix_new                  (void);

LeTensor * le_matrix_new_copy             (LeTensor     *another);

LeTensor * le_matrix_new_from_data        (unsigned      height,
                                           unsigned      width,
                                           const float  *data);

void       le_matrix_set_element          (LeTensor     *matrix,
                                           unsigned      y,
                                           unsigned      x,
                                           float         value);

void       le_tensor_subtract             (LeTensor     *a,
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

void       le_tensor_free                 (LeTensor     *matrix);

/** @note: Temporary */
#include <stdio.h>

void       le_matrix_print                (LeTensor     *matrix,
                                           FILE         *stream);

/** @note: Inner product of two column vectors */
float      le_dot_product                 (LeTensor     *a,
                                           LeTensor     *b);

float      le_rbf                         (LeTensor     *a,
                                           LeTensor     *b,
                                           float         sigma);

#endif
