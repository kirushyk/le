/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_MODEL_H__
#define __LE_MODEL_H__

#include "leobject.h"
#include "letensor.h"

typedef struct LeModel
{
    LeObject parent;
} LeModel;

typedef struct LeModelClass
{
    LeClass parent;
    LeMatrix * (*predict)(LeModel *model, LeMatrix *x);
    float (*train_iteration)(LeModel *model);
} LeModelClass;

typedef struct LeModel LeModel;

void       le_model_construct       (LeModel  *model);

LeMatrix * le_model_predict         (LeModel  *model,
                                     LeMatrix *x);

float      le_model_train_iteration (LeModel  *model);

void       le_model_free            (LeModel  *model);

#endif
