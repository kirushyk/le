/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEKNN_H__
#define __LEKNN_H__

#include <glib.h>
#include "lemodel.h"
#include <le/tensors/letensor.h>

G_BEGIN_DECLS

G_DECLARE_FINAL_TYPE (LeKNN, le_knn, LE, KNN, LeModel);


/** @note: Lazy algorithm which just remembers training set */
// typedef struct LeKNN LeKNN;

#define LE_KNN(obj) ((LeKNN *)(obj))

LeKNN *                 le_knn_new                        (void);

void                    le_knn_train                      (LeKNN *                  knn,
                                                           LeTensor *               x,
                                                           LeTensor *               y,
                                                           unsigned                 k);

LeTensor *              le_knn_predict                    (LeModel *                model,
                                                           const LeTensor *         x);

// void                    le_knn_free                       (LeKNN *                  knn);

G_END_DECLS

#endif
