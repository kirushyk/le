/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "bgd"

#include "lebgd.h"
#include <le/letensor.h>
#include <le/lelog.h>
#include <stdlib.h>
#include <stdio.h>

struct LeBGD
{
    LeOptimizer parent;
    float learning_rate;
};

typedef struct LeBGDClass
{
    LeOptimizerClass parent;
} LeBGDClass;

LeBGDClass le_bgd_class;

static void
le_bgd_step(LeOptimizer *optimizer)
{
    LeBGD *self = (LeBGD *)optimizer;
    LeList *parameters_iterator;
    LeList *gradients_iterator;

    LE_INFO("Step");

    for (parameters_iterator = optimizer->parameters, gradients_iterator = optimizer->gradients;
         parameters_iterator && gradients_iterator;
         parameters_iterator = parameters_iterator->next, gradients_iterator = gradients_iterator->next)
    {
        LeTensor *parameter = (LeTensor *)parameters_iterator->data;
        LeTensor *gradients = (LeTensor *)gradients_iterator->data;
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
le_bgd_class_ensure_init(void)
{
    static int le_bgd_class_initialized = 0;

    if (!le_bgd_class_initialized)
    {
        le_bgd_class.parent.step =
            (void (*)(LeOptimizer *))le_bgd_step;
        le_bgd_class_initialized = 1;
    }
}

void
le_bgd_construct(LeBGD *self)
{
    le_optimizer_construct((LeOptimizer *)self);
    le_bgd_class_ensure_init();
    ((LeObject *)self)->klass = (LeClass *)&le_bgd_class;
}

LeBGD *
le_bgd_new(LeList *parameters, float learning_rate)
{
    LeBGD *self = malloc(sizeof(LeBGD));
    le_bgd_construct(self);
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
le_bgd_free(LeBGD *optimizer)
{
    free(optimizer);
}
