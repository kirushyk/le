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

typedef struct _LeSGD
{
  LeOptimizer parent;
} LeSGD;

typedef struct _LeSGDPrivate
{
  unsigned k;
  
  LeTensor *input;
  LeTensor *output;

  size_t batch_size;
  unsigned example_index;
  float momentum_rate;
  GList *momenta;
} LeSGDPrivate;

static void le_sgd_class_init (LeSGDClass * klass);
static void le_sgd_init (LeSGD * self);
G_DEFINE_FINAL_TYPE_WITH_PRIVATE (LeSGD, le_sgd, LE_TYPE_OPTIMIZER);

static void
le_sgd_dispose (GObject * object)
{
  LeSGD *self = LE_SGD (object);
  g_assert_nonnull (self);
  LeSGDPrivate *priv = le_sgd_get_instance_private (self);
  g_assert_nonnull (priv);
  g_list_free_full (priv->momenta, (GDestroyNotify)le_tensor_free);
  G_OBJECT_CLASS (le_sgd_parent_class)->dispose (object);
}

static void
le_sgd_finalize (GObject * object)
{
}

void le_sgd_step (LeOptimizer * optimizer);

void le_sgd_epoch (LeOptimizer * optimizer);

static void
le_sgd_class_init (LeSGDClass * klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_sgd_dispose;
  G_OBJECT_CLASS (klass)->finalize = le_sgd_finalize;
  LE_OPTIMIZER_CLASS (klass)->step = le_sgd_step;
  LE_OPTIMIZER_CLASS (klass)->epoch = le_sgd_epoch;
}

static void
le_sgd_init (LeSGD * self)
{
  LeSGDPrivate *priv = le_sgd_get_instance_private (self);
  // LE_OPTIMIZER(self)->model = model;
  // LE_OPTIMIZER(self)->step = 0;
  // LE_OPTIMIZER(self)->epoch = 0;
  // LE_OPTIMIZER(self)->parameters = le_model_get_parameters(LE_OPTIMIZER(self)->model);
  // LE_OPTIMIZER(self)->learning_rate = learning_rate;

  priv->input = NULL;
  priv->output = NULL;
  priv->batch_size = 1;
  priv->example_index = 0;
  priv->momenta = NULL;
  priv->momentum_rate = 0.8;
}

GList *
le_sgd_init_momenta (GList * gradients)
{
  GList *momentum_list = NULL;
  for (GList *gradients_iterator = gradients; 
        gradients_iterator;
        gradients_iterator = gradients_iterator->next)
  {
    LeTensor *momentum = le_tensor_new_zeros_like(LE_TENSOR(gradients_iterator->data));
    momentum_list = g_list_append(momentum_list, momentum);
  }
  return momentum_list;
}

void
le_sgd_step(LeOptimizer *optimizer)
{
    LeSGD *self = LE_SGD(optimizer);
    LeSGDPrivate *priv = le_sgd_get_instance_private (self);
    GList *parameters_iterator;
    GList *gradients_iterator;
    GList *momentum_iterator;

    // LE_INFO("Epoch %u Step %u", optimizer->epoch, optimizer->step);
    
    unsigned num_examples = le_matrix_get_width(priv->input);

    size_t batch_size = priv->example_index + priv->batch_size < num_examples ? priv->batch_size : num_examples - priv->example_index;

    LeTensor *input = le_matrix_get_columns_copy(priv->input, priv->example_index, batch_size);
    LeTensor *output = le_matrix_get_columns_copy(priv->output, priv->example_index, batch_size);

    le_optimizer_get_gradients (optimizer) = le_model_get_gradients(optimizer->model, input, output);

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
        
    if (priv->momenta == NULL)
    {
        priv->momenta = le_sgd_init_momenta(le_optimizer_get_gradients (optimizer));
    }

    le_tensor_free(output);
    le_tensor_free(input);

    for (parameters_iterator = le_optimizer_get_gradients (parameters),
            gradients_iterator = le_optimizer_get_gradients (optimizer),
            momentum_iterator = priv->momenta;
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
        le_tensor_mul(momentum, priv->momentum_rate);
        le_tensor_mul(gradient, 1.0f - priv->momentum_rate);
        le_tensor_add(momentum, gradient);
        le_tensor_sub_scaled(parameter, optimizer->learning_rate, momentum);
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
    
    g_list_free_full (le_optimizer_get_gradients (optimizer), (GDestroyNotify)le_tensor_free);
    le_optimizer_get_gradients (optimizer) = NULL;
    
    optimizer->step++;
    priv->example_index += batch_size;
    if (priv->example_index >= num_examples) {
        priv->example_index = 0;
    }
    // if (optimizer->step * priv->batch_size >= num_examples)
    // {
    //     optimizer->epoch++;
    // }
}

void
le_sgd_epoch(LeOptimizer *optimizer)
{
    LeSGD *self = LE_SGD(optimizer);
    unsigned num_examples = le_matrix_get_width(priv->input);
    for (unsigned i = 0; i < num_examples; i++)
    {
        le_sgd_step(optimizer);
    }
}

LeSGD *
le_sgd_new(LeModel *model, LeTensor *input, LeTensor *output, size_t batch_size, float learning_rate, float momentum)
{
  assert(model);
  LeSGD *self = g_object_new (le_sgd_get_type (), NULL);
  g_assert_cmpfloat (learning_rate, >, 0.0f);

  LeSGDPrivate *priv = le_sgd_get_instance_private (self);
  g_assert_nonnull (priv);
  
  // LE_OPTIMIZER(self)->model = model;
  // LE_OPTIMIZER(self)->parameters = le_model_get_parameters(LE_OPTIMIZER(self)->model);

  priv->input = input;
  priv->output = output;
  priv->batch_size = batch_size;
  priv->example_index = 0;
  priv->momenta = NULL;
  priv->momentum_rate = momentum;

  return self;
}
