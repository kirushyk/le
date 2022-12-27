/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <le/le.h>

#ifndef __LECUDA_H__
#define __LECUDA_H__

LeTensor * le_cuda_matrix_new_product (const LeTensor * a,
                                       bool             transpose_a,
                                       const LeTensor * b,
                                       bool             transpose_b);

#endif
