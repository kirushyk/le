/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information.
 
   Vanilla Batch Gradient Descent Optimization Algorithm
 
 */

#ifndef __LEBGD_H__
#define __LEBGD_H__

#include "leoptimizer.h"
#include <le/lelist.h>

typedef struct LeBGD LeBGD;

LeBGD * le_bgd_new  (LeList *parameters,
                     float   learning_rate);

void    le_bgd_free (LeBGD  *optimizer);

#endif
