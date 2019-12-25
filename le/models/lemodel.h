/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_MODEL_H__
#define __LE_MODEL_H__

#include "../leobject.h"
#include "../letensor.h"
#include "../lelist.h"

typedef struct LeModel
{
    LeObject  parent;
    LeList   *parameters;
} LeModel;

#define LE_MODEL(obj) ((LeModel *)(obj))

typedef struct LeModelClass
{
    LeClass parent;
    LeTensor * (*predict)(LeModel *model, LeTensor *x);
    LeList * (*get_gradients)(LeModel *model, LeTensor *x, LeTensor *y);
    float (*train_iteration)(LeModel *model);
} LeModelClass;

void       le_model_construct       (LeModel  *model);

LeTensor * le_model_predict         (LeModel  *model,
                                     LeTensor *x);

LeList *   le_model_get_gradients   (LeModel  *model,
                                     LeTensor *x,
                                     LeTensor *y);

float      le_model_train_iteration (LeModel  *model);

LeList *   le_model_get_parameters  (LeModel  *model);

void       le_model_free            (LeModel  *model);

#endif
