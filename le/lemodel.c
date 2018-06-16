/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lemodel.h"

LeModelClass le_model_class;

void le_model_construct(LeModel *model)
{
    ((LeObject *)model)->klass = (LeClass *)&le_model_class;
}

LeMatrix *
le_model_predict(LeModel *self, LeMatrix *x)
{
    return ((LeModelClass *)((LeObject *)self)->klass)->predict(self, x);
}

void
le_model_free(LeModel *self)
{
    
}
