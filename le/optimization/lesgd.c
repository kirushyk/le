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

    size_t batch_size;
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

void
le_sgd_step(LeOptimizer *optimizer)
{
    LeSGD *self = LE_SGD(optimizer);
    LeList *parameters_iterator;
    LeList *gradients_iterator;
    LeList *momentum_iterator;

    LE_INFO("Epoch %u Step %u", optimizer->epoch, optimizer->step);
    
    unsigned num_examples = le_matrix_get_width(self->input);

    unsigned example_index = (optimizer->step * self->batch_size) % num_examples;

    size_t batch_size = example_index + self->batch_size < num_examples ? self->batch_size : num_examples - example_index;

    LeTensor *input = le_matrix_get_columns_copy(self->input, example_index, batch_size);
    LeTensor *output = le_matrix_get_columns_copy(self->output, example_index, batch_size);

    optimizer->gradients = le_model_get_gradients(optimizer->model, input, output);

    // LE_INFO("Input %s:\n%s", le_shape_to_cstr(input->shape), le_tensor_to_cstr(input));
    // LeTensorStats input_stats = le_tensor_get_stats(input);
    // LE_INFO("Input stats:\tmin: %f\tmax: %f\tmean: %f\tdeviation: %f\t nans: %u\t zeros: %u",
    //     input_stats.min, input_stats.max, input_stats.mean, input_stats.deviation,
    //     input_stats.nans, input_stats.zeros);
        
    // LE_INFO("Output %s:\n%s", le_shape_to_cstr(output->shape), le_tensor_to_cstr(output));
    // LeTensorStats output_stats = le_tensor_get_stats(output);
    // LE_INFO("Output stats:\tmin: %f\tmax: %f\tmean: %f\tdeviation: %f\t nans: %u\t zeros: %u",
    //     output_stats.min, output_stats.max, output_stats.mean, output_stats.deviation,
    //     output_stats.nans, output_stats.zeros);
        
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
        // LE_INFO("Parameter %s:\n%s", le_shape_to_cstr(parameter->shape), le_tensor_to_cstr(parameter));
        // LeTensorStats parameter_stats = le_tensor_get_stats(parameter);
        // LE_INFO("Parameter stats:\tmin: %f\tmax: %f\tmean: %f\tdeviation: %f\t nans: %u\t zeros: %u",
        //     parameter_stats.min, parameter_stats.max, parameter_stats.mean, parameter_stats.deviation,
        //     parameter_stats.nans, parameter_stats.zeros);
        LeTensor *gradient = (LeTensor *)gradients_iterator->data;
        // LE_INFO("Gradient %s:\n%s", le_shape_to_cstr(gradient->shape), le_tensor_to_cstr(gradient));
        // LeTensorStats gradient_stats = le_tensor_get_stats(gradient);
        // LE_INFO("Gradient stats:\tmin: %f\ttmax: %f\tmean: %f\tdeviation: %f\t nans: %u\t zeros: %u",
        //     gradient_stats.min, gradient_stats.max, gradient_stats.mean, gradient_stats.deviation,
        //     gradient_stats.nans, gradient_stats.zeros);
        LeTensor *momentum = LE_TENSOR(momentum_iterator->data);
        le_tensor_add(momentum, gradient);
        le_tensor_sub_scaled(parameter, optimizer->learning_rate, momentum);
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
    
    le_list_free(optimizer->gradients, LE_FUNCTION(le_tensor_free));
    optimizer->gradients = NULL;
    
    optimizer->step++;
    // if (optimizer->step * self->batch_size >= num_examples)
    // {
    //     optimizer->epoch++;
    // }
}

void
le_sgd_epoch(LeOptimizer *optimizer)
{
    LeSGD *self = LE_SGD(optimizer);
    unsigned num_examples = le_matrix_get_width(self->input);
    for (unsigned i = 0; i < num_examples; i++)
    {
        le_sgd_step(optimizer);
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
        klass.parent.epoch =
            (void (*)(LeOptimizer *))le_sgd_epoch;
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
le_sgd_new(LeModel *model, LeTensor *input, LeTensor *output, size_t batch_size, float learning_rate, float momentum)
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
    self->batch_size = batch_size;
    self->momenta = NULL;
    self->momentum_rate = momentum;
    return self;
}

void
le_sgd_free(LeSGD *self)
{
    le_list_free(self->momenta, LE_FUNCTION(le_tensor_free));
    free(self);
}
