/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEKNN_H__
#define __LEKNN_H__

#include <le/lemacros.h>
#include <le/tensors/letensor.h>

LE_BEGIN_DECLS

/** @note: Lazy algorithm which just remembers training set */
typedef struct LeKNN LeKNN;

#define LE_KNN(obj) ((LeKNN *)(obj))

LeKNN * le_knn_new     (void);

void                    le_knn_train                      (LeKNN *                  knn,
                                                           LeTensor *               x,
                                                           LeTensor *               y,
                                                           unsigned                 k);

void                    le_knn_free                       (LeKNN *                  knn);

LE_END_DECLS

#endif
