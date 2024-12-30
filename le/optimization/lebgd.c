/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "bgd"

#include "lebgd.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <le/tensors/letensor.h>
#include <le/tensors/letensor-imp.h>
#include <le/lelog.h>

typedef struct _LeBGD
{
  LeOptimizer parent;
} LeBGD;

typedef struct _LeBGDPrivate
{
  unsigned k;
  
  const LeTensor *input;
  const LeTensor *output;
} LeBGDPrivate;

static void le_bgd_class_init (LeBGDClass * klass);
static void le_bgd_init (LeBGD * self);
G_DEFINE_FINAL_TYPE_WITH_PRIVATE (LeBGD, le_bgd, le_optimizer_get_type ());

static void
le_bgd_dispose (GObject * object)
{
  LeBGD *self = LE_BGD (object);
  g_assert_nonnull (self);
  LeBGDPrivate *priv = le_bgd_get_instance_private (self);
  g_assert_nonnull (priv);
//   LeTensor *input;
//   LeTensor *output;
  G_OBJECT_CLASS (le_bgd_parent_class)->dispose (object);
}

static void
le_bgd_finalize (GObject * object)
{
}

void le_bgd_step (LeOptimizer * optimizer);

void le_bgd_epoch (LeOptimizer * optimizer);

static void
le_bgd_class_init (LeBGDClass * klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_bgd_dispose;
  G_OBJECT_CLASS (klass)->finalize = le_bgd_finalize;
  LE_OPTIMIZER_CLASS (klass)->step = le_bgd_step;
  LE_OPTIMIZER_CLASS (klass)->epoch = le_bgd_epoch;
}

static void
le_bgd_init (LeBGD * self)
{
  LeBGDPrivate *priv = le_bgd_get_instance_private (self);
  // LE_OPTIMIZER(self)->model = model;
  // LE_OPTIMIZER(self)->step = 0;
  // LE_OPTIMIZER(self)->epoch = 0;
  // LE_OPTIMIZER(self)->parameters = le_model_get_parameters(LE_OPTIMIZER(self)->model);
  // LE_OPTIMIZER(self)->learning_rate = learning_rate;

  priv->input = NULL;
  priv->output = NULL;
}

void
le_bgd_step(LeOptimizer *optimizer)
{
    LeBGD *self = LE_BGD (optimizer);
    LeBGDPrivate *priv = le_bgd_get_instance_private (self);
    GList *parameters_iterator;
    GList *gradients_iterator;

    LE_INFO("Step");

    gfloat learning_rate = le_optimizer_get_learning_rate (optimizer);

    GList *gradients = NULL;
    bool own_gradients = false;

    LeModel *model = le_optimizer_get_model (optimizer);
    if (model)
    {
        gradients = le_model_get_gradients(model, priv->input, priv->output);
        own_gradients = true;
    }
    else if (le_optimizer_get_gradients (optimizer))
    {
        gradients = le_optimizer_get_gradients (optimizer);
    }
    

    for (parameters_iterator = le_optimizer_get_parameters (optimizer), gradients_iterator = gradients;
         parameters_iterator && gradients_iterator;
         parameters_iterator = parameters_iterator->next, gradients_iterator = gradients_iterator->next)
    {
        LeTensor *parameter = (LeTensor *)parameters_iterator->data;
        LE_INFO("Parameter %s:\n%s", le_shape_to_cstr(parameter->shape), le_tensor_to_cstr(parameter));
        LeTensor *gradient = (LeTensor *)gradients_iterator->data;
        LE_INFO("Gradient %s:\n%s", le_shape_to_cstr(gradient->shape), le_tensor_to_cstr(gradient));
        le_tensor_sub_scaled_f32(parameter, learning_rate, gradient);
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
        g_list_free_full (gradients, (GDestroyNotify)le_tensor_free);
    }

    // LE_OPTIMIZER(self)->step++;
    // LE_OPTIMIZER(self)->epoch++;
}

void
le_bgd_epoch(LeOptimizer *optimizer)
{
    le_bgd_step(optimizer);
}

// void
// le_bgd_class_ensure_init(void)
// {
//     static bool initialized = false;

//     if (!initialized)
//     {
//         klass.parent.step =
//             (void (*)(LeOptimizer *))le_bgd_step;
//         klass.parent.epoch =
//             (void (*)(LeOptimizer *))le_bgd_epoch; /// @note: epoch == step for BGD
//         initialized = 1;
//     }
// }

// void
// le_bgd_construct(LeBGD *self)
// {
//     le_optimizer_construct((LeOptimizer *)self);
//     le_bgd_class_ensure_init();
//     ((GObject *)self)->klass = (GObjectClass *)&klass;
// }

LeBGD * 
le_bgd_new_simple(GList *parameters, GList *gradients, gfloat learning_rate)
{
  assert(parameters);
  assert(gradients);

  LeBGD *self = g_object_new (le_bgd_get_type (), NULL);
  LeBGDPrivate *priv = le_bgd_get_instance_private (self);
  // le_bgd_construct(self);
  if (learning_rate <= 0.0f)
  {
      LE_WARNING("Learning rate = %f", learning_rate);
  }
  // LE_OPTIMIZER(self)->model = NULL;
  // LE_OPTIMIZER(self)->step = 0;
  // LE_OPTIMIZER(self)->epoch = 0;
  // LE_OPTIMIZER(self)->parameters = parameters;
  // LE_OPTIMIZER(self)->gradients = gradients;
  // LE_OPTIMIZER(self)->learning_rate = learning_rate;

  priv->input = NULL;
  priv->output = NULL;
  return self;
}

LeBGD *
le_bgd_new(LeModel * model, const LeTensor * input, const LeTensor * output, gfloat learning_rate)
{
  assert(model);

  LeBGD *self = g_object_new (le_bgd_get_type (), NULL);
  LeBGDPrivate *priv = le_bgd_get_instance_private (self);
  if (learning_rate <= 0.0f)
  {
    LE_WARNING ("Learning rate = %f", learning_rate);
  }
  le_optimizer_set_learning_rate (LE_OPTIMIZER (self), learning_rate);
  le_optimizer_set_model (LE_OPTIMIZER (self), model);

  priv->input = input;
  priv->output = output;
  return self;
}
