/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "ledenselayer.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <le/tensors/lematrix.h>

struct _LeDenseLayer
{
    LeLayer parent;
};

typedef struct _LeDenseLayerPrivate
{
    LeTensor *w;
    LeTensor *b;
} LeDenseLayerPrivate;

static void le_dense_layer_class_init (LeDenseLayerClass * klass);
static void le_dense_layer_init (LeDenseLayer * self);
G_DEFINE_FINAL_TYPE_WITH_PRIVATE (LeDenseLayer, le_dense_layer, LE_TYPE_LAYER);

LeTensor *
le_dense_layer_forward_prop(LeLayer *layer, LeTensor *input)
{
  assert(layer);
  assert(input);
  
  LeDenseLayer *self = LE_DENSE_LAYER(layer);
  LeDenseLayerPrivate *priv = le_dense_layer_get_instance_private (self);
  g_assert_nonnull (priv);
  g_assert_nonnull (priv->w);

  LeTensor *output = le_matrix_new_product (priv->w, input);
  
  if (priv->b)
  {
    le_matrix_add(output, priv->b);
  }
  
  return output;
}

LeTensor *
le_dense_layer_backward_prop(LeLayer *layer, LeTensor *cached_input, LeTensor *cached_output, LeTensor *output_gradient, GList **parameters_gradient)
{
  assert(layer);
  assert(output_gradient);
  
  LeDenseLayer *self = LE_DENSE_LAYER(layer);
  LeDenseLayerPrivate *priv = le_dense_layer_get_instance_private (self);
  g_assert_nonnull (priv);


  assert(priv->w);

  LeTensor *input_gradient = le_matrix_new_product_full(priv->w, true, output_gradient, false);

  if (parameters_gradient)
  {
    assert(cached_input);

    LeTensor *h = le_tensor_new_copy(output_gradient);
    unsigned examples_count = le_matrix_get_width(h);
    le_tensor_mul(h, 1.0f / examples_count);
    LeTensor *dw = le_matrix_new_product_full(h, false, cached_input, true);
    LeTensor *db = le_matrix_new_sum(h, 1);
    le_tensor_free(h);
    *parameters_gradient = g_list_append(*parameters_gradient, db);
    *parameters_gradient = g_list_append(*parameters_gradient, dw);
  }

  return input_gradient;
}

static void
le_dense_layer_dispose (GObject * object)
{
  G_OBJECT_CLASS (le_dense_layer_parent_class)->dispose (object);
}

static void
le_dense_layer_finalize (GObject * object)
{
}

LeShape *
le_dense_layer_get_output_shape(LeLayer *layer)
{
    LeDenseLayer *self = LE_DENSE_LAYER(layer);
    LeDenseLayerPrivate *priv = le_dense_layer_get_instance_private (self);
    g_assert_nonnull (priv);

    return le_shape_new(2, le_matrix_get_height(priv->b), 0);
}

const char *
le_dense_layer_get_description(LeLayer *self)
{
    static const char *description = "Fully Connected Layer";
    return description;
}

static void
le_dense_layer_class_init (LeDenseLayerClass * klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_dense_layer_dispose;
  G_OBJECT_CLASS (klass)->finalize = le_dense_layer_finalize;
  LE_LAYER_CLASS (klass)->forward_prop = le_dense_layer_forward_prop;
  LE_LAYER_CLASS (klass)->backward_prop = le_dense_layer_backward_prop;
  LE_LAYER_CLASS (klass)->get_output_shape = le_dense_layer_get_output_shape;
  LE_LAYER_CLASS (klass)->get_description = le_dense_layer_get_description;
}

static void
le_dense_layer_init (LeDenseLayer * self)
{

}

LeDenseLayer *
le_dense_layer_new(const char *name, unsigned inputs, unsigned units)
{
  LeDenseLayer *self = g_object_new (le_dense_layer_get_type (), NULL);
  LeDenseLayerPrivate *priv = le_dense_layer_get_instance_private (self);
  g_assert_nonnull (priv);

  // le_layer_construct(LE_LAYER(self), name);
  // le_dense_layer_class_ensure_init();
  // G_OBJECT_GET_CLASS(self) = G_OBJECT_CLASS(&klass);
  priv->w = le_matrix_new_rand_f32(LE_DISTRIBUTION_NORMAL, units, inputs);
  /// @todo: Optimize
  gfloat variance = sqrtf(1.0f / inputs);
  le_tensor_mul(priv->w, variance);
  priv->b = le_matrix_new_zeros(LE_TYPE_FLOAT32, units, 1);
  le_layer_append_parameter(LE_LAYER(self), priv->w);
  le_layer_append_parameter(LE_LAYER(self), priv->b);
  return self;
}
