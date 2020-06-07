/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEOPTIMIZER_H__
#define __LEOPTIMIZER_H__

#include "../leobject.h"
#include <le/lelist.h>
#include <le/lemacros.h>
#include <le/models/lemodel.h>

LE_BEGIN_DECLS

/// Base class for all optimizers
typedef struct LeOptimizer
{
    LeObject                 parent;

    LeModel                 *model;
    LeList                  *parameters;
    LeList                  *gradients;
    float                    learning_rate;
    unsigned                 step;
    unsigned                 epoch;
} LeOptimizer;

#define LE_OPTIMIZER(obj) ((LeOptimizer *)(obj))

typedef struct LeOptimizerClass
{
    LeClass parent;
    void (*step)(LeOptimizer *model);
} LeOptimizerClass;

#define LE_OPTIMIZER_CLASS(klass) ((LeOptimizerClass *)(klass))
#define LE_OPTIMIZER_GET_CLASS(obj) (LE_OPTIMIZER_CLASS(LE_OBJECT_GET_CLASS(obj)))

void               le_optimizer_construct                  (LeOptimizer *           optimizer);

void               le_optimizer_step                       (LeOptimizer *           optimizer);

void               le_optimizer_free                       (LeOptimizer *           optimizer);

LE_END_DECLS

#endif
