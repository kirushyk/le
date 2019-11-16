/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leoptimizer.h"
#include <stdlib.h>

LeOptimizerClass le_optimizer_class;

void
le_optimizer_construct(LeOptimizer *optimizer)
{
    ((LeObject *)optimizer)->klass = (LeClass *)&le_optimizer_class;
}

void
le_optimizer_free(LeOptimizer *optimizer)
{
    free(optimizer);
}
