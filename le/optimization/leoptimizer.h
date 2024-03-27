/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEOPTIMIZER_H__
#define __LEOPTIMIZER_H__

#include <glib.h>
#include <le/models/lemodel.h>

G_BEGIN_DECLS

#define LE_TYPE_OPTIMIZER (le_optimizer_get_type ())
G_DECLARE_DERIVABLE_TYPE (LeOptimizer, le_optimizer, LE, OPTIMIZER, GObject);

/// Base class for all optimizers
// typedef struct LeOptimizer
// {
//     GObject                 parent;

//     LeModel                 *model;
//     GList                   *parameters;
//     GList                   *gradients;
//     float                    learning_rate;
//     unsigned                 step;
//     unsigned                 epoch;
// } LeOptimizer;

// #define LE_OPTIMIZER(obj) ((LeOptimizer *)(obj))

struct _LeOptimizerClass
{
    GObjectClass parent;
    void (*step)(LeOptimizer *optimizer);
    void (*epoch)(LeOptimizer *optimizer);
};

// #define LE_OPTIMIZER_CLASS(klass) ((LeOptimizerClass *)(klass))
// #define LE_OPTIMIZER_GET_CLASS(obj) (LE_OPTIMIZER_CLASS(G_OBJECT_GET_CLASS(obj)))

void               le_optimizer_step                       (LeOptimizer *           optimizer);

void               le_optimizer_epoch                      (LeOptimizer *           optimizer);

G_END_DECLS

#endif
