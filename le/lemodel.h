/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_MODEL_H__
#   define __LE_MODEL_H__
#   include "leobject.h"
#   include "lematrix.h"

typedef struct LeModel LeModel;

LeModel *  le_model_new     (void);

LeMatrix * le_model_predict (LeModel  *model,
                             LeMatrix *x);

void       le_model_free    (LeModel  *model);

#endif
