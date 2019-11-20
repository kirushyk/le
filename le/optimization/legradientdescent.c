/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "legradientdescent.h"
#include <stdlib.h>

struct LeGradientDescent
{
    LeOptimizer parent;
    float learning_rate;
};

typedef struct LeGradientDescentClass
{
    LeOptimizerClass parent;
} LeGradientDescentClass;

LeGradientDescentClass le_gradient_descent_class;

static void
le_gradient_descent_step(LeOptimizer *optimizer)
{
}

void
le_gradient_descent_class_ensure_init(void)
{
    static int le_gradient_descent_class_initialized = 0;

    if (!le_gradient_descent_class_initialized)
    {
        le_gradient_descent_class.parent.step =
            (void (*)(LeOptimizer *))le_gradient_descent_step;
        le_gradient_descent_class_initialized = 1;
    }
}

void
le_gradient_descent_construct(LeGradientDescent *self)
{
    le_optimizer_construct((LeOptimizer *)self);
    le_gradient_descent_class_ensure_init();
    ((LeObject *)self)->klass = (LeClass *)&le_gradient_descent_class;
}

LeGradientDescent *
le_gradient_descent_new(LeList *parameters, float learning_rate)
{
    LeGradientDescent *self = malloc(sizeof(LeGradientDescent));
    le_gradient_descent_construct(self);
    self->learning_rate = learning_rate;
    return self;
}

void
le_gradient_descent_free(LeGradientDescent *optimizer)
{
    free(optimizer);
}
