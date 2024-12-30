/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lelayer.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <le/tensors/lematrix.h>
#include <le/tensors/letensor-imp.h>

typedef struct _LeLayerPrivate
{
  GList      *parameters;
  const char *name;
} LeLayerPrivate;

static void le_layer_class_init (LeLayerClass * klass);
static void le_layer_init (LeLayer * self);
G_DEFINE_TYPE_WITH_PRIVATE (LeLayer, le_layer, G_TYPE_OBJECT);

static void
le_layer_dispose (GObject * object)
{
  G_OBJECT_CLASS (le_layer_parent_class)->dispose (object);
}

static void
le_layer_finalize (GObject * object)
{
   
}

static void
le_layer_class_init (LeLayerClass * klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_layer_dispose;
  G_OBJECT_CLASS (klass)->finalize = le_layer_finalize;
}

static void
le_layer_init (LeLayer * self)
{
  LeLayerPrivate *priv = le_layer_get_instance_private (self);
  priv->parameters = NULL;
  priv->name = NULL;
}

// void
// le_layer_construct(LeLayer *self, const char *name)
// {
//     assert(self);
    
//     self->parameters = NULL;
//     self->name = g_strdup(name);
// }

const gchar *
le_layer_get_name (const LeLayer * self)
{
  LeLayerPrivate *priv = le_layer_get_instance_private ((LeLayer *)self);
  return priv->name;
}

LeTensor *
le_layer_forward_prop(LeLayer *self, LeTensor *input)
{
    assert(self);
    LeLayerClass *klass = LE_LAYER_GET_CLASS(self);
    assert(klass);
    g_assert_nonnull (klass->forward_prop);

    return klass->forward_prop(self, input);
}

GList *
le_layer_get_parameters(LeLayer *self)
{
  assert(self);
  LeLayerPrivate *priv = le_layer_get_instance_private (self);
  g_assert_nonnull (priv);

  return priv->parameters;
}

unsigned     
le_layer_get_parameters_count(LeLayer * self)
{
  g_assert_nonnull (self);
  LeLayerPrivate *priv = le_layer_get_instance_private (self);
  g_assert_nonnull (priv);
  unsigned count = 0;
  for (GList *current = priv->parameters; current != NULL; current = current->next)
  {
      count += le_shape_get_elements_count(LE_TENSOR(current->data)->shape);
  }
  return count;
}

void
le_layer_append_parameter(LeLayer *self, LeTensor *parameter)
{
  assert(self);
  assert(parameter);
  LeLayerPrivate *priv = le_layer_get_instance_private (self);
  g_assert_nonnull (priv);
    
  priv->parameters = g_list_append(priv->parameters, parameter);
}

LeTensor * 
le_layer_backward_prop(LeLayer *self, LeTensor *cached_input, LeTensor *cached_output, LeTensor *output_gradient, GList **parameters_gradient)
{
    assert(self);
    LeLayerClass *klass = LE_LAYER_GET_CLASS(self);
    assert(klass);
    assert(klass->backward_prop);
    assert(output_gradient);
    
    return klass->backward_prop(self, cached_input, cached_output, output_gradient, parameters_gradient);
}

LeShape *
le_layer_get_output_shape(LeLayer *self)
{
    assert(self);
    LeLayerClass *klass = LE_LAYER_GET_CLASS(self);
    assert(klass);
    assert(klass->get_output_shape);

    return klass->get_output_shape(self);
}

const char *
le_layer_get_description(LeLayer *self)
{
    assert(self);
    LeLayerClass *klass = LE_LAYER_GET_CLASS(self);
    assert(klass);
    assert(klass->get_description);

    return klass->get_description(self);
}
