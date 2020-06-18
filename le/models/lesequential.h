/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_SEQUENTIAL_H__
#define __LE_SEQUENTIAL_H__

#include "../lemacros.h"
#include <le/tensors/letensor.h>
#include "../leloss.h"
#include "lemodel.h"
#include "layers/lelayer.h"

LE_BEGIN_DECLS

/// Sequential model is a plain stack of layers where each layer has exactly one input and one output
typedef struct LeSequential LeSequential;

#define LE_SEQUENTIAL(a) ((LeSequential *)(a))

LeSequential *          le_sequential_new                  (void);

void                    le_sequential_add                  (LeSequential *          model,
                                                            LeLayer *               layer);

void                    le_sequential_set_loss             (LeSequential *          model,
                                                            LeLoss                  loss);

LeTensor *              le_sequential_predict              (LeSequential *          model,
                                                            const LeTensor *        x);

LeList *                le_sequential_get_gradients        (LeSequential           *model,
                                                            const LeTensor         *x, 
                                                            const LeTensor         *y);

LeList *                le_sequential_estimate_gradients   (LeSequential           *model,
                                                            const LeTensor         *x, 
                                                            const LeTensor         *y);

void                    le_sequential_to_dot               (LeSequential *          model,
                                                            const char *            filename);

void                    le_sequential_free                 (LeSequential *          model);

LE_END_DECLS

#endif
