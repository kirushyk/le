/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information.
 
   Vanilla Stochastic Gradient Descent Optimization Algorithm
 
 */

#ifndef __LEBGD_H__
#define __LEBGD_H__

#include "leoptimizer.h"
#include <le/lelist.h>

typedef struct LeSGD LeSGD;

#define LE_SGD(o) ((LeSGD *)(o))

LeSGD * le_sgd_new  (LeList *parameters,
                     float   learning_rate);

void    le_sgd_free (LeSGD  *optimizer);

#endif
