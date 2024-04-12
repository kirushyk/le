/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leoptimizer.h"
#include <stdlib.h>
#include <assert.h>

// void
// le_optimizer_construct(LeOptimizer *self)
// {
//     G_OBJECT_GET_CLASS(self) = G_OBJECT_CLASS(&klass);
//     self->model = NULL;
//     self->parameters = NULL;
//     self->gradients = NULL;
// }
typedef struct _LeOptimizerPrivate
{
  LeModel  *model;
  GList    *parameters;
  GList    *gradients;
  gfloat     learning_rate;
  unsigned  step;
  unsigned  epoch;
} LeOptimizerPrivate;

static void le_optimizer_class_init (LeOptimizerClass * klass);
static void le_optimizer_init (LeOptimizer * self);
G_DEFINE_TYPE_WITH_PRIVATE (LeOptimizer, le_optimizer, G_TYPE_OBJECT);

static void
le_optimizer_dispose (GObject * object)
{
  G_OBJECT_CLASS (le_optimizer_parent_class)->dispose (object);
}

static void
le_optimizer_finalize (GObject * object)
{
   
}

static void
le_optimizer_class_init (LeOptimizerClass * klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_optimizer_dispose;
  G_OBJECT_CLASS (klass)->finalize = le_optimizer_finalize;
  klass->step = NULL;
  klass->epoch = NULL;
}

static void
le_optimizer_init (LeOptimizer * self)
{
  LeOptimizerPrivate *priv = le_optimizer_get_instance_private (self);
  priv->model = NULL;
  priv->step = 0;
  priv->epoch = 0;
  priv->learning_rate = 0;
  priv->parameters = NULL;
}

void
le_optimizer_step(LeOptimizer *self)
{
  assert(self);
  assert(G_OBJECT_GET_CLASS(self));
  assert(LE_OPTIMIZER_GET_CLASS(self)->step);
  
  LE_OPTIMIZER_GET_CLASS(self)->step(self);
}

void
le_optimizer_epoch(LeOptimizer *self)
{
  assert(self);
  assert(G_OBJECT_GET_CLASS(self));
  assert(LE_OPTIMIZER_GET_CLASS(self)->epoch);
  
  LE_OPTIMIZER_GET_CLASS(self)->epoch(self);
}

GList *
le_optimizer_get_parameters (LeOptimizer * self)
{
  g_assert_nonnull (self);
  LeOptimizerPrivate *priv = le_optimizer_get_instance_private (self);
  g_assert_nonnull (priv);
  return priv->parameters;
}

GList *
le_optimizer_get_gradients (LeOptimizer * self)
{
  g_assert_nonnull (self);
  LeOptimizerPrivate *priv = le_optimizer_get_instance_private (self);
  g_assert_nonnull (priv);
  return priv->gradients;
}

void
le_optimizer_set_gradients (LeOptimizer * self, GList * gradients)
{
  g_assert_nonnull (self);
  LeOptimizerPrivate *priv = le_optimizer_get_instance_private (self);
  g_assert_nonnull (priv);
  priv->gradients = gradients;
}

gfloat 
le_optimizer_get_learning_rate (LeOptimizer * self)
{
  g_assert_nonnull (self);
  LeOptimizerPrivate *priv = le_optimizer_get_instance_private (self);
  g_assert_nonnull (priv);
  g_assert_cmpfloat (priv->learning_rate, >, 0);
  return priv->learning_rate;
}

void
le_optimizer_set_learning_rate (LeOptimizer * self, gfloat learning_rate)
{
  g_assert_nonnull (self);
  g_assert_cmpfloat (learning_rate, >, 0);
  LeOptimizerPrivate *priv = le_optimizer_get_instance_private (self);
  g_assert_nonnull (priv);
  priv->learning_rate = learning_rate;
}

LeModel *
le_optimizer_get_model (LeOptimizer * self)
{
  g_assert_nonnull (self);
  LeOptimizerPrivate *priv = le_optimizer_get_instance_private (self);
  g_assert_nonnull (priv);
  return priv->model;
}
