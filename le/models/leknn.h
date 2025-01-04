/* Le - Machine Learning Library
 *
 * K-Nearest Neighbors (KNN) - Lazy classifier that simply remembers training data for further predictions.
 *
 * Copyright (c) 2020 Kyrylo Polezhaiev. All rights reserved.
 * Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_MODELS_LEKNN_H__
#define __LE_MODELS_LEKNN_H__

#include <glib.h>
#include <le/tensors/letensor.h>
#include "lemodel.h"

G_BEGIN_DECLS

G_DECLARE_FINAL_TYPE (LeKNN, le_knn, LE, KNN, LeModel);

#define LE_KNN(obj) ((LeKNN *)(obj))

LeKNN *    le_knn_new     (void);

void       le_knn_train   (LeKNN *                  knn,
                           LeTensor *               x,
                           LeTensor *               y,
                           unsigned                 k);

LeTensor * le_knn_predict (LeModel *                model,
                           const LeTensor *         x);

G_END_DECLS

#endif
