/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information.
 
  Stochastic Gradient Descent with Momentum Optimization Algorithm
 
 */

#ifndef __LESGD_H__
#define __LESGD_H__

#include <glib.h>
#include "leoptimizer.h"

G_BEGIN_DECLS

G_DECLARE_FINAL_TYPE (LeSGD, le_sgd, LE, SGD, LeOptimizer);
// typedef struct LeSGD LeSGD;

// #define LE_SGD(o) ((LeSGD *)(o))

LeSGD *            le_sgd_new                              (LeModel *               model,
                                                            LeTensor *              input,
                                                            LeTensor *              output,
                                                            gsize                  batch_size,
                                                            gfloat                   learning_rate,
                                                            gfloat                   momentum);

void               le_sgd_step                             (LeOptimizer *           optimizer);

void               le_sgd_epoch                            (LeOptimizer *           optimizer);

// void               le_sgd_free                             (LeSGD *                 optimizer);

G_END_DECLS

#endif
