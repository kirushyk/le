/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEGRADIENTDESCENT_H__
#define __LEGRADIENTDESCENT_H__

#include "leoptimizer.h"

typedef struct LeGradientDescent
{
    LeOptimizer parent;
} LeGradientDescent;

typedef struct LeGradientDescentClass
{
    LeOptimizerClass parent;
} LeGradientDescentClass;

LeGradientDescent * le_gradient_descent_new  (void);

void                le_gradient_descent_free (LeGradientDescent *optimizer);

#endif
