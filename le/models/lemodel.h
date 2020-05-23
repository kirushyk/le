/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_MODEL_H__
#define __LE_MODEL_H__

#include "../lemacros.h"
#include "../leobject.h"
#include "../letensor.h"
#include "../lelist.h"

LE_BEGIN_DECLS

typedef struct LeModel
{
    LeObject  parent;
    LeList   *parameters;
} LeModel;

#define LE_MODEL(obj) ((LeModel *)(obj))

typedef struct LeModelClass
{
    LeClass parent;
    LeTensor * (*predict)         (LeModel *model, const LeTensor *x);
    LeList *   (*get_gradients)   (LeModel *model, const LeTensor *x, const LeTensor *y);
    float      (*train_iteration) (LeModel *model);
} LeModelClass;

#define LE_MODEL_CLASS(klass) ((LeModelClass *)(klass))
#define LE_MODEL_GET_CLASS(obj) (LE_MODEL_CLASS(LE_OBJECT_GET_CLASS(obj)))

void       le_model_construct        (LeModel        *model);

/** @note: This function is to be used by instances of subclasses of LeModel
    to list its trainable parameters */
void       le_model_append_parameter (LeModel        *model,
                                      LeTensor       *parameter);

LeTensor * le_model_predict          (LeModel        *model,
                                      const LeTensor *x);

LeList *   le_model_get_gradients    (LeModel        *model,
                                      LeTensor       *x,
                                      LeTensor       *y);

float      le_model_train_iteration  (LeModel        *model);

LeList *   le_model_get_parameters   (LeModel        *model);

void       le_model_free             (LeModel        *model);

LE_END_DECLS

#endif
