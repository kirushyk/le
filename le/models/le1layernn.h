/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE1LAYERNN_H__
#define __LE1LAYERNN_H__

#include <le/tensors/letensor.h>
#include <glib.h>

G_BEGIN_DECLS

/// @note: Temporary Model needed to develop MLP and SGD. To be removed later.
typedef struct Le1LayerNN Le1LayerNN;

Le1LayerNN * le_1_layer_nn_new   (void);

typedef struct Le1LayerNNTrainingOptions
{
    float            learning_rate;
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

G_END_DECLS

#endif
