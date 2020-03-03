/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information.
 
   Vanilla Batch Gradient Descent Optimization Algorithm
 
 */

#ifndef __LEBGD_H__
#define __LEBGD_H__

#include "leoptimizer.h"
#include <le/lelist.h>
#include <le/models/lemodel.h>

typedef struct LeBGD LeBGD;

#define LE_BGD(o) ((LeBGD *)(o))

LeBGD * le_bgd_new_simple (LeList   *parameters,
                           LeList   *gradients,
                           float     learning_rate);

LeBGD * le_bgd_new        (LeModel  *model,
                           LeTensor *input,
                           LeTensor *output,
                           float     learning_rate);

void    le_bgd_free       (LeBGD    *optimizer);

#endif
