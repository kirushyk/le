/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "bgd"

#include "lebgd.h"
#include <assert.h>
#include <le/letensor.h>
#include <le/letensor-imp.h>
#include <le/lelog.h>
#include <stdlib.h>
#include <stdio.h>

struct LeBGD
{
    LeOptimizer parent;
    
    unsigned iteration;
    LeModel *model;
    LeTensor *input;
    LeTensor *output;
    float learning_rate;
};

typedef struct LeBGDClass
{
    LeOptimizerClass parent;
} LeBGDClass;

static LeBGDClass klass;

static void
le_bgd_step(LeOptimizer *optimizer)
{
    LeBGD *self = (LeBGD *)optimizer;
    LeList *parameters_iterator;
    LeList *gradients_iterator;

    LE_INFO("Step");

    LeList *gradients = NULL;
    bool own_gradients = false;

    if (self->model)
    {
        gradients = le_model_get_gradients(self->model, self->input, self->output);
        own_gradients = true;
    }
    else if (optimizer->gradients)
    {
        gradients = optimizer->gradients;
    }
    

    for (parameters_iterator = optimizer->parameters, gradients_iterator = gradients;
         parameters_iterator && gradients_iterator;
         parameters_iterator = parameters_iterator->next, gradients_iterator = gradients_iterator->next)
    {
        LeTensor *parameter = (LeTensor *)parameters_iterator->data;
        LE_INFO("Parameter %s:\n%s", le_shape_to_cstr(parameter->shape), le_tensor_to_cstr(parameter));
        LeTensor *gradient = (LeTensor *)gradients_iterator->data;
        LE_INFO("Gradient %s:\n%s", le_shape_to_cstr(gradient->shape), le_tensor_to_cstr(gradient));
        le_tensor_sub_scaled(parameter, self->learning_rate, gradient);
        LeTensorStats gradient_stats = le_tensor_get_stats(gradient);
        LE_INFO("Gradient stats:\n\tmin: %f\n\tmax: %f\n\tmean: %f\n\tdeviation: %f", gradient_stats.min, gradient_stats.max, gradient_stats.mean, gradient_stats.deviation);
    }

    if (parameters_iterator)
    {
        LE_WARNING("Some gradients missing");
    }

    if (gradients_iterator)
    {
        LE_WARNING("Extra gradients passed");
    }

    if (own_gradients)
    {
        le_list_foreach(gradients, (LeFunction)le_tensor_free);
    }
}

void
le_bgd_class_ensure_init(void)
{
    static bool initialized = false;

    if (!initialized)
    {
        klass.parent.step =
            (void (*)(LeOptimizer *))le_bgd_step;
        initialized = 1;
    }
}

void
le_bgd_construct(LeBGD *self)
{
    le_optimizer_construct((LeOptimizer *)self);
    le_bgd_class_ensure_init();
    ((LeObject *)self)->klass = (LeClass *)&klass;
}

LeBGD * 
le_bgd_new_simple(LeList *parameters, LeList *gradients, float learning_rate)
{
    assert(parameters);
    assert(gradients);

    LeBGD *self = malloc(sizeof(LeBGD));
    le_bgd_construct(self);
    if (learning_rate <= 0.0f)
    {
        LE_WARNING("Learning rate = %f", learning_rate);
    }
    LE_OPTIMIZER(self)->parameters = parameters;
    LE_OPTIMIZER(self)->gradients = gradients;
    self->learning_rate = learning_rate;

    self->iteration = 0;
    self->model = NULL;
    self->input = NULL;
    self->output = NULL;
    return self;
}

LeBGD *
le_bgd_new(LeModel *model, LeTensor *input, LeTensor *output, float learning_rate)
{
    assert(model);

    LeBGD *self = malloc(sizeof(LeBGD));
    le_bgd_construct(self);
    if (learning_rate <= 0.0f)
    {
        LE_WARNING("Learning rate = %f", learning_rate);
    }
    LE_OPTIMIZER(self)->parameters = le_model_get_parameters(model);
    LE_OPTIMIZER(self)->gradients = NULL;
    self->learning_rate = learning_rate;

    self->iteration = 0;
    self->model = model;
    self->input = input;
    self->output = output;
    return self;
}

void
le_bgd_free(LeBGD *optimizer)
{
    free(optimizer);
}
