/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEOPTIMIZER_H__
#define __LEOPTIMIZER_H__

#include "../leobject.h"

typedef struct LeOptimizer
{
    LeObject parent;
} LeOptimizer;

typedef struct LeOptimizerClass
{
    LeClass parent;
} LeOptimizerClass;

void le_optimizer_construct (LeOptimizer *optimizer);

void le_optimizer_free      (LeOptimizer *optimizer);

#endif

