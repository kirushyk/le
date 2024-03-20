/* Copyright (c) 2024 Kyrylo Polezhaiev. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information.
 
  "Adam: A Method for Stochastic Optimization".
 
 */

#ifndef __LE__OPTIMIZATION__ADAM_H__
#define __LE__OPTIMIZATION__ADAM_H__

#include <le/lemacros.h>
#include <le/lelist.h>
#include "leoptimizer.h"

LE_BEGIN_DECLS

typedef struct LeAdam LeAdam;

#define LE_ADAM(o) ((LeAdam *)(o))

LeAdam * le_adam_new   (LeModel *     model,
                        LeTensor *    input,
                        LeTensor *    output,
                        size_t        batch_size,
                        float         learning_rate,
                        float         beta_1,
                        float         beta_2);

void     le_adam_step  (LeOptimizer * optimizer);

void     le_adam_epoch (LeOptimizer * optimizer);

void     le_adam_free  (LeAdam *      optimizer);

LE_END_DECLS

#endif
