/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __BACKENDS_ACCELERATE_LEACCELERATE_H__
#define __BACKENDS_ACCELERATE_LEACCELERATE_H__

#include <le/le.h>

G_BEGIN_DECLS

LeTensor * le_accelerate_matrix_new_product         (const LeTensor *a,
                                                     gboolean        transpose_a,
                                                     const LeTensor *b,
                                                     gboolean        transpose_b);

void       le_accelerate_tensor_apply_sigmoid       (LeTensor       *tensor);

void       le_accelerate_tensor_apply_sigmoid_prime (LeTensor       *tensor);

gfloat     le_accelerate_rbf                        (const LeTensor *a,
                                                     const LeTensor *b,
                                                     gfloat          sigma);

gfloat     le_accelerate_dot_product                (const LeTensor *a,
                                                     const LeTensor *b);

G_END_DECLS

#endif
