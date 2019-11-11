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
    LeTensor * (*predict)(LeModel *model, LeTensor *x);
    float (*train_iteration)(LeModel *model);
} LeModelClass;

void       le_model_construct       (LeModel  *model);

LeTensor * le_model_predict         (LeModel  *model,
                                     LeTensor *x);

float      le_model_train_iteration (LeModel  *model);

void       le_model_free            (LeModel  *model);

#endif
