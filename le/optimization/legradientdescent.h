/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEGRADIENTDESCENT_H__
#define __LEGRADIENTDESCENT_H__

#include "leoptimizer.h"
#include <le/lelist.h>

typedef struct LeGradientDescent LeGradientDescent;

LeGradientDescent * le_gradient_descent_new  (LeList            *parameters,
                                              float              learning_rate);

void                le_gradient_descent_free (LeGradientDescent *optimizer);

#endif
