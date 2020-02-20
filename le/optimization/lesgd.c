/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "sgd"

#include "lesgd.h"
#include <le/letensor.h>
#include <le/letensor-imp.h>
#include <le/lelog.h>
#include <stdlib.h>
#include <stdio.h>

struct LeSGD
{
    LeOptimizer parent;
    float learning_rate;
};

typedef struct LeSGDClass
{
    LeOptimizerClass parent;
} LeSGDClass;

static LeSGDClass klass;

static void
le_sgd_step(LeOptimizer *optimizer)
{
    LeSGD *self = (LeSGD *)optimizer;
    LeList *parameters_iterator;
    LeList *gradients_iterator;

    LE_INFO("Step");

    for (parameters_iterator = optimizer->parameters, gradients_iterator = optimizer->gradients;
         parameters_iterator && gradients_iterator;
         parameters_iterator = parameters_iterator->next, gradients_iterator = gradients_iterator->next)
    {
        LeTensor *parameter = (LeTensor *)parameters_iterator->data;
        LE_INFO("Parameter %s:\n%s", le_shape_to_cstr(parameter->shape), le_tensor_to_cstr(parameter));
        LeTensor *gradients = (LeTensor *)gradients_iterator->data;
        LE_INFO("Gradient %s:\n%s", le_shape_to_cstr(gradients->shape), le_tensor_to_cstr(gradients));
        le_tensor_subtract_scaled(parameter, self->learning_rate, gradients);
    }

    if (parameters_iterator)
    {
        LE_WARNING("Some gradients missing");
    }

    if (gradients_iterator)
    {
        LE_WARNING("Extra gradients passed");
    }
}

void
le_sgd_class_ensure_init(void)
{
    static bool initialized = false;

    if (!initialized)
    {
        klass.parent.step =
            (void (*)(LeOptimizer *))le_sgd_step;
        initialized = 1;
    }
}

void
le_sgd_construct(LeSGD *self)
{
    le_optimizer_construct((LeOptimizer *)self);
    le_sgd_class_ensure_init();
    ((LeObject *)self)->klass = (LeClass *)&klass;
}

LeSGD *
le_sgd_new(LeList *parameters, float learning_rate)
{
    LeSGD *self = malloc(sizeof(LeSGD));
    le_sgd_construct(self);
    if (parameters == NULL)
    {
        LE_WARNING("Empty list of parameters to optimize passed");
    }
    if (learning_rate <= 0.0f)
    {
        LE_WARNING("Learning rate = %f", learning_rate);
    }
    LE_OPTIMIZER(self)->parameters = parameters;
    self->learning_rate = learning_rate;
    return self;
}

void
le_sgd_free(LeSGD *optimizer)
{
    free(optimizer);
}
