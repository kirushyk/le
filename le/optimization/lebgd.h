/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information.
 
   Vanilla Batch Gradient Descent Optimization Algorithm
 
 */

#ifndef __LEBGD_H__
#define __LEBGD_H__

#include <glib.h>
#include "leoptimizer.h"

G_BEGIN_DECLS

G_DECLARE_FINAL_TYPE (LeBGD, le_bgd, LE, BGD, LeOptimizer);

// typedef struct LeBGD LeBGD;

// #define LE_BGD(o) ((LeBGD *)(o))

LeBGD *            le_bgd_new_simple                       (GList *                parameters,
                                                            GList *                gradients,
                                                            float                   learning_rate);

LeBGD *            le_bgd_new                              (LeModel *               model,
                                                            const LeTensor *        input,
                                                            const LeTensor *        output,
                                                            float                   learning_rate);

/// @note: le_optimizer_step virtual method will call this function if optimizer subclass is BGD.
/// Exposed to speed-up C++ bindings.
void               le_bgd_step                             (LeOptimizer *           optimizer);

void               le_bgd_epoch                            (LeOptimizer *           optimizer);

// void               g_object_unref                             (LeBGD *                 optimizer);

G_END_DECLS

#endif
