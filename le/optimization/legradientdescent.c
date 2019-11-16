/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "legradientdescent.h"
#include <stdlib.h>

LeGradientDescent *
le_gradient_descent_new()
{
    LeGradientDescent *self = malloc(sizeof(LeGradientDescent));
    return self;
}

void
le_gradient_descent_free(LeGradientDescent *optimizer)
{
    free(optimizer);
}
