/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information.
 
   Vanilla Stochastic Gradient Descent Optimization Algorithm
 
 */

#ifndef __LESGD_H__
#define __LESGD_H__

#include "leoptimizer.h"
#include <le/lelist.h>
#include <le/models/lemodel.h>

typedef struct LeSGD LeSGD;

#define LE_SGD(o) ((LeSGD *)(o))

LeSGD * le_sgd_new  (LeModel  *model,
                     LeTensor *input,
                     LeTensor *output,
                     float     learning_rate);

void    le_sgd_free (LeSGD  *optimizer);

#endif
