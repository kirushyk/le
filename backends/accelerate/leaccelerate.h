/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEACCELERATE_H__
#define __LEACCELERATE_H__

#include <le/le.h>

LE_BEGIN_DECLS

LeTensor * le_accelerate_matrix_new_product         (const LeTensor *a,
                                                     bool            transpose_a,
                                                     const LeTensor *b,
                                                     bool            transpose_b);

void       le_accelerate_tensor_apply_sigmoid       (LeTensor *tensor);

void       le_accelerate_tensor_apply_sigmoid_prime (LeTensor *tensor);

float      le_accelerate_rbf                        (const LeTensor *a,
                                                     const LeTensor *b,
                                                     float           sigma);

float      le_accelerate_dot_product                (const LeTensor *a,
                                                     const LeTensor *b);

LE_END_DECLS

#endif
