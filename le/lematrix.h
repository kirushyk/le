/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "letensor.h"

#ifndef __LEMATRIX_H__
#define __LEMATRIX_H__

float      le_matrix_at                   (LeTensor     *matrix,
                                           unsigned      y,
                                           unsigned      x);

#endif
