/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lemodel.h"
#include <assert.h>
#include <stdlib.h>

LeModelClass le_model_class;

void le_model_construct(LeModel *self)
{
    LE_OBJECT_CLASS(self) = (LeClass *)&le_model_class;
    self->parameters = NULL;
}

LeTensor *
le_model_predict(LeModel *self, LeTensor *x)
{
    assert(self);
    assert(LE_OBJECT_CLASS(self));
    assert(((LeModelClass *)LE_OBJECT_CLASS(self))->predict);
    
    return ((LeModelClass *)LE_OBJECT_CLASS(self))->predict(self, x);
}

LeList *
le_model_get_gradients(LeModel *self, LeTensor *x, LeTensor *y)
{
    assert(self);
    assert(LE_OBJECT_CLASS(self));
    
    if (((LeModelClass *)LE_OBJECT_CLASS(self))->get_gradients == NULL)
    {
        return NULL;
    };
    
    return ((LeModelClass *)LE_OBJECT_CLASS(self))->get_gradients(self, x, y);
}

float
le_model_train_iteration(LeModel *self)
{
    return ((LeModelClass *)LE_OBJECT_CLASS(self))->train_iteration(self);
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