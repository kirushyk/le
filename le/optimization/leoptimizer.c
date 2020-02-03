/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leoptimizer.h"
#include <stdlib.h>
#include <assert.h>

static LeOptimizerClass klass;

void
le_optimizer_construct(LeOptimizer *self)
{
    ((LeObject *)self)->klass = (LeClass *)&klass;
}

void
le_optimizer_step(LeOptimizer *self)
{
    assert(self);
    assert(((LeObject *)self)->klass);
    assert(((LeOptimizerClass *)((LeObject *)self)->klass)->step);
    
    ((LeOptimizerClass *)((LeObject *)self)->klass)->step(self);
}

void
le_optimizer_free(LeOptimizer *self)
{
    free(self);
}
