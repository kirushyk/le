/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEOPTIMIZER_H__
#define __LEOPTIMIZER_H__

#include "../leobject.h"
#include <le/lelist.h>

typedef struct LeOptimizer
{
    LeObject parent;
    LeList *parameters;
    LeList *gradients;
} LeOptimizer;

#define LE_OPTIMIZER(obj) ((LeOptimizer *)(obj))

typedef struct LeOptimizerClass
{
    LeClass parent;
    void (*step)(LeOptimizer *model);
} LeOptimizerClass;

void le_optimizer_construct (LeOptimizer *optimizer);

void le_optimizer_step      (LeOptimizer *optimizer);

void le_optimizer_free      (LeOptimizer *optimizer);

#endif
