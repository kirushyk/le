/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <le/le.h>

#ifndef __LEACCELERATE_H__
#define __LEACCELERATE_H__

LeTensor * le_accelerate_matrix_new_product   (LeTensor *a,
                                               bool      transpose_a,
                                               LeTensor *b,
                                               bool      transpose_b);

void       le_accelerate_tensor_apply_sigmoid (LeTensor     *tensor);


#endif
