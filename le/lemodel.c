/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lemodel.h"
#include <assert.h>
#include <stdlib.h>

LeModelClass le_model_class;

void le_model_construct(LeModel *model)
{
    ((LeObject *)model)->klass = (LeClass *)&le_model_class;
    model->parameters = NULL;
}

LeTensor *
le_model_predict(LeModel *self, LeTensor *x)
{
    assert(self);
    assert(((LeObject *)self)->klass);
    assert(((LeModelClass *)((LeObject *)self)->klass)->predict);
    
    return ((LeModelClass *)((LeObject *)self)->klass)->predict(self, x);
}

LeList *
le_model_get_gradients(LeModel *self, LeTensor *x, LeTensor *y)
{
    assert(self);
    assert(((LeObject *)self)->klass);
    
    if (((LeModelClass *)((LeObject *)self)->klass)->get_gradients == NULL)
    {
        return NULL;
    };
    
    return ((LeModelClass *)((LeObject *)self)->klass)->get_gradients(self, x, y);
}

float
le_model_train_iteration(LeModel *self)
{
    return ((LeModelClass *)((LeObject *)self)->klass)->train_iteration(self);
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
