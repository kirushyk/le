/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_SEQUENTIAL_H__
#define __LE_SEQUENTIAL_H__

#include "../letensor.h"
#include "../leloss.h"
#include "lemodel.h"
#include "layers/lelayer.h"

typedef struct LeSequential LeSequential;

#define LE_SEQUENTIAL(a) ((LeSequential *)(a))

LeSequential * le_sequential_new      (void);

void           le_sequential_add      (LeSequential *model,
                                       LeLayer      *layer);

void           le_sequential_set_loss (LeSequential *model,
                                       LeLoss        loss);

LeTensor *     le_sequential_predict  (LeSequential *model,
                                       LeTensor     *x);

void           le_sequential_to_dot   (LeSequential *model,
                                       const char   *filename);

void           le_sequential_free     (LeSequential *model);

#endif
