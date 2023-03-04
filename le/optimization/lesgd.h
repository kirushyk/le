/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information.
 
  Stochastic Gradient Descent with Momentum Optimization Algorithm
 
 */

#ifndef __LESGD_H__
#define __LESGD_H__

#include <le/lemacros.h>
#include <le/lelist.h>
#include "leoptimizer.h"

LE_BEGIN_DECLS

typedef struct LeSGD LeSGD;

#define LE_SGD(o) ((LeSGD *)(o))

LeSGD *            le_sgd_new                              (LeModel *               model,
                                                            LeTensor *              input,
                                                            LeTensor *              output,
                                                            size_t                  batch_size,
                                                            float                   learning_rate,
                                                            float                   momentum);

void               le_sgd_step                             (LeOptimizer *           optimizer);

void               le_sgd_epoch                            (LeOptimizer *           optimizer);

void               le_sgd_free                             (LeSGD *                 optimizer);

LE_END_DECLS

#endif
