/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "sgd"

#include "lesgd.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <le/letensor.h>
#include <le/letensor-imp.h>
#include <le/lelog.h>
#include <le/lematrix.h>

struct LeSGD
{
    LeOptimizer parent;
    float learning_rate;

    unsigned iteration;
    LeModel *model;
    LeTensor *input;
    LeTensor *output;
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
    
    unsigned num_examples = le_matrix_get_width(self->input);
    unsigned example_index = self->iteration % num_examples;
    LeTensor *input = le_matrix_get_column(self->input, example_index);
    LeTensor *output = le_matrix_get_column(self->output, example_index);

    optimizer->gradients = le_model_get_gradients(self->model, input, output);

    le_tensor_free(output);
    le_tensor_free(input);

    for (parameters_iterator = optimizer->parameters, gradients_iterator = optimizer->gradients;
         parameters_iterator && gradients_iterator;
         parameters_iterator = parameters_iterator->next, gradients_iterator = gradients_iterator->next)
    {
        LeTensor *parameter = (LeTensor *)parameters_iterator->data;
        LE_INFO("Parameter %s:\n%s", le_shape_to_cstr(parameter->shape), le_tensor_to_cstr(parameter));
        LeTensor *gradients = (LeTensor *)gradients_iterator->data;
        LE_INFO("Gradient %s:\n%s", le_shape_to_cstr(gradients->shape), le_tensor_to_cstr(gradients));
        le_tensor_subtract_scaled(parameter, self->learning_rate, gradients);
        LeTensorStats gradient_stats = le_tensor_get_stats(gradients);
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
    
    le_list_foreach(optimizer->gradients, (LeFunction)le_tensor_free);
    
    self->iteration++;
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
le_sgd_new(LeModel *model, LeTensor *input, LeTensor *output, float learning_rate)
{
    assert(model);
    LeSGD *self = malloc(sizeof(LeSGD));
    le_sgd_construct(self);
    if (learning_rate <= 0.0f)
    {
        LE_WARNING("Learning rate = %f", learning_rate);
    }
    LE_OPTIMIZER(self)->parameters = le_model_get_parameters(model);
    self->learning_rate = learning_rate;

    self->iteration = 0;
    self->model = model;
    self->input = input;
    self->output = output;
    return self;
}

void
le_sgd_free(LeSGD *optimizer)
{
    free(optimizer);
}
