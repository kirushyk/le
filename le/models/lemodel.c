/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lemodel.h"
#include <assert.h>
#include <stdlib.h>
#include "lelog.h"

#define DEFAULT_LOG_CATEGORY "model"

LeModelClass le_model_class;

void
le_model_construct(LeModel *self)
{
    LE_OBJECT_GET_CLASS(self) = (LeClass *)&le_model_class;
    self->parameters = NULL;
}

void
le_model_append_parameter(LeModel *self, LeTensor *parameter)
{
    self->parameters = le_list_append(self->parameters, parameter);
}

LeTensor *
le_model_predict(LeModel *self, LeTensor *x)
{
    assert(self);
    assert(LE_OBJECT_GET_CLASS(self));
    assert(LE_MODEL_GET_CLASS(self)->predict);
    
    return LE_MODEL_GET_CLASS(self)->predict(self, x);
}

LeList *
le_model_get_gradients(LeModel *self, LeTensor *x, LeTensor *y)
{
    assert(self);
    assert(LE_OBJECT_GET_CLASS(self));
    
    if (LE_MODEL_GET_CLASS(self)->get_gradients == NULL)
    {
        LE_WARNING("`get_gradients` virtual function is not set in subclass");
        return NULL;
    };
    
    return LE_MODEL_GET_CLASS(self)->get_gradients(self, x, y);
}

float
le_model_train_iteration(LeModel *self)
{
    return LE_MODEL_GET_CLASS(self)->train_iteration(self);
}

LeList *
le_model_get_parameters(LeModel *self)
{
    return self->parameters;
}

void
le_model_free(LeModel *self)
{
    free(self);
}
