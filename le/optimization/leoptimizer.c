/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leoptimizer.h"
#include <stdlib.h>
#include <assert.h>

static LeOptimizerClass klass;

void
le_optimizer_construct(LeOptimizer *self)
{
    LE_OBJECT_GET_CLASS(self) = LE_CLASS(&klass);
    self->model = NULL;
    self->parameters = NULL;
    self->gradients = NULL;
}

void
le_optimizer_step(LeOptimizer *self)
{
    assert(self);
    assert(LE_OBJECT_GET_CLASS(self));
    assert(LE_OPTIMIZER_GET_CLASS(self)->step);
    
    LE_OPTIMIZER_GET_CLASS(self)->step(self);
}

void
le_optimizer_free(LeOptimizer *self)
{
    free(self);
}
