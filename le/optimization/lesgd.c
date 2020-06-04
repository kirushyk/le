/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lesgd.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <le/tensors/letensor.h>
#include <le/tensors/letensor-imp.h>
#include <le/lelog.h>
#include <le/tensors/lematrix.h>

#define DEFAULT_LOG_CATEGORY "sgd"

struct LeSGD
{
    LeOptimizer parent;

    LeTensor *input;
    LeTensor *output;

    float momentum_rate;
    LeList *momenta;
};

typedef struct LeSGDClass
{
    LeOptimizerClass parent;
} LeSGDClass;

static LeSGDClass klass;

LeList *
le_sgd_init_momenta(LeList *gradients)
{
    LeList *momentum_list = NULL;
    for (LeList *gradients_iterator = gradients; 
         gradients_iterator;
         gradients_iterator = gradients_iterator->next)
    {
        LeTensor *momentum = le_tensor_new_zeros_like(LE_TENSOR(gradients_iterator->data));
        momentum_list = le_list_append(momentum_list, momentum);
    }
    return momentum_list;
}

static void
le_sgd_step(LeOptimizer *optimizer)
{
    LeSGD *self = (LeSGD *)optimizer;
    LeList *parameters_iterator;
    LeList *gradients_iterator;
    LeList *momentum_iterator;

    LE_INFO("Step");
    
    unsigned num_examples = le_matrix_get_width(self->input);
    unsigned example_index = LE_OPTIMIZER(self)->step % num_examples;
    LeTensor *input = le_matrix_get_column(self->input, example_index);
    LeTensor *output = le_matrix_get_column(self->output, example_index);

    optimizer->gradients = le_model_get_gradients(LE_OPTIMIZER(self)->model, input, output);

    if (self->momenta == NULL)
    {
        self->momenta = le_sgd_init_momenta(optimizer->gradients);
    }

    le_tensor_free(output);
    le_tensor_free(input);

    for (parameters_iterator = optimizer->parameters,
            gradients_iterator = optimizer->gradients,
            momentum_iterator = self->momenta;
         parameters_iterator &&
            gradients_iterator &&
            momentum_iterator;
         parameters_iterator = parameters_iterator->next,
            gradients_iterator = gradients_iterator->next,
            momentum_iterator = momentum_iterator->next)
    {
        LeTensor *parameter = (LeTensor *)parameters_iterator->data;
        LE_INFO("Parameter %s:\n%s", le_shape_to_cstr(parameter->shape), le_tensor_to_cstr(parameter));
        LeTensorStats parameter_stats = le_tensor_get_stats(parameter);
        LE_INFO("Parameter stats:\n\tmin: %f\n\tmax: %f\n\tmean: %f\n\tdeviation: %f", parameter_stats.min, parameter_stats.max, parameter_stats.mean, parameter_stats.deviation);
        LeTensor *gradient = (LeTensor *)gradients_iterator->data;
        LE_INFO("Gradient %s:\n%s", le_shape_to_cstr(gradient->shape), le_tensor_to_cstr(gradient));
        LeTensorStats gradient_stats = le_tensor_get_stats(gradient);
        LE_INFO("Gradient stats:\n\tmin: %f\n\tmax: %f\n\tmean: %f\n\tdeviation: %f", gradient_stats.min, gradient_stats.max, gradient_stats.mean, gradient_stats.deviation);
        LeTensor *momentum = LE_TENSOR(momentum_iterator->data);
        le_tensor_add(momentum, gradient);
        le_tensor_sub_scaled_f32(parameter, optimizer->learning_rate, momentum);
        le_tensor_mul(momentum, self->momentum_rate);
    }

    if (parameters_iterator)
    {
        LE_WARNING("Some gradients missing");
    }

    if (gradients_iterator)
    {
        LE_WARNING("Extra gradients passed");
    }
    
    if (momentum_iterator)
    {
        LE_WARNING("Extra momenta passed");
    }
    
    le_list_foreach(optimizer->gradients, (LeFunction)le_tensor_free);
    
    LE_OPTIMIZER(self)->step++;
    if (LE_OPTIMIZER(self)->step % num_examples == 0)
    {
        LE_OPTIMIZER(self)->epoch++;
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
le_sgd_new(LeModel *model, LeTensor *input, LeTensor *output, float learning_rate, float momentum)
{
    assert(model);
    LeSGD *self = malloc(sizeof(LeSGD));
    le_sgd_construct(self);
    if (learning_rate <= 0.0f)
    {
        LE_WARNING("Learning rate = %f", learning_rate);
    }
    LE_OPTIMIZER(self)->model = model;
    LE_OPTIMIZER(self)->step = 0;
    LE_OPTIMIZER(self)->epoch = 0;
    LE_OPTIMIZER(self)->parameters = le_model_get_parameters(LE_OPTIMIZER(self)->model);
    LE_OPTIMIZER(self)->learning_rate = learning_rate;

    self->input = input;
    self->output = output;
    self->momenta = NULL;
    self->momentum_rate = momentum;
    return self;
}

void
le_sgd_free(LeSGD *optimizer)
{
    le_list_foreach(optimizer->momenta, LE_FUNCTION(le_tensor_free));
    free(optimizer);
}
