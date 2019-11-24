/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE1LAYERNN_H__
#define __LE1LAYERNN_H__

#include "letensor.h"

typedef struct Le1LayerNN Le1LayerNN;

Le1LayerNN * le_1_layer_nn_new   (void);

typedef struct Le1LayerNNTrainingOptions
{
    float            alpha;
    unsigned         max_iterations;
} Le1LayerNNTrainingOptions;

void         le_1_layer_nn_init           (Le1LayerNN                *classifier,
                                           unsigned                   features_count,
                                           unsigned                   classes_count);

void         le_1_layer_nn_train          (Le1LayerNN                *classifier,
                                           LeTensor                  *x_train,
                                           LeTensor                  *y_train,
                                           Le1LayerNNTrainingOptions  options);

void         le_1_layer_nn_free           (Le1LayerNN                *classifier);

#endif
