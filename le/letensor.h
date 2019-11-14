/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdbool.h>
#include "letype.h"
#include "leshape.h"

#ifndef __LETENSOR_H__
#define __LETENSOR_H__

typedef struct LeTensor LeTensor;

LeTensor * le_tensor_new                  (void);

LeTensor * le_tensor_new_from_data        (LeType        element_type,
                                           LeShape      *shape,
                                           void         *data);

LeTensor * le_tensor_new_copy             (LeTensor     *another);

bool       le_tensor_reshape              (LeTensor     *tensor,
                                           unsigned      num_dimensions,
                                           ...);

LeTensor * le_tensor_pick                 (LeTensor     *another,
                                           uint32_t      index);

LeTensor * le_tensor_pick_copy            (LeTensor     *another,
                                           uint32_t      index);

uint8_t    le_tensor_at                   (LeTensor     *tensor,
                                           uint32_t      index);

void       le_tensor_subtract             (LeTensor     *a,
                                           LeTensor     *b);

void       le_tensor_multiply_by_scalar   (LeTensor     *a,
                                           float         b);

void       le_tensor_add_scalar           (LeTensor     *a,
                                           float         b);

float      le_tensor_sum                  (LeTensor     *matrix);

void       le_tensor_apply_sigmoid        (LeTensor     *matrix);

void       le_tensor_apply_greater_than   (LeTensor     *matrix,
                                           float         scalar);

void       le_tensor_apply_svm_prediction (LeTensor     *matrix);

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
