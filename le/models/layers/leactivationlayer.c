/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leactivationlayer.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <le/lelog.h>
#include <le/tensors/lematrix.h>
#include <le/tensors/letensor-imp.h>

#define DEFAULT_LOG_CATEGORY "activation-layer"

struct _LeActivationLayer
{
  LeLayer parent;
};

// typedef struct LeActivationLayerClass
// {
//     LeLayerClass parent;
    
// } LeActivationLayerClass;

typedef struct _LeActivationLayerPrivate
{
    LeActivation activation;
} LeActivationLayerPrivate;

static void le_activation_layer_class_init (LeActivationLayerClass * klass);
static void le_activation_layer_init (LeActivationLayer * self);
G_DEFINE_FINAL_TYPE_WITH_PRIVATE (LeActivationLayer, le_activation_layer, LE_TYPE_LAYER);

static void
le_activation_layer_dispose (GObject * object)
{
  G_OBJECT_CLASS (le_activation_layer_parent_class)->dispose (object);
}

static void
le_activation_layer_finalize (GObject * object)
{
}

LeTensor * le_activation_layer_forward_prop(LeLayer *layer, LeTensor *input);
LeTensor * le_activation_layer_backward_prop(LeLayer *layer, LeTensor *cached_input, LeTensor *cached_output, LeTensor *output_gradient, GList **parameters_gradient);
LeShape * le_activation_layer_get_output_shape(LeLayer *layer);
const char * le_activation_layer_get_description(LeLayer *layer);

static void
le_activation_layer_class_init (LeActivationLayerClass * klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_activation_layer_dispose;
  G_OBJECT_CLASS (klass)->finalize = le_activation_layer_finalize;
  // LeTensor * (*forward_prop)(LeLayer *self, LeTensor *x);
  // LeTensor * (*backward_prop)(LeLayer *self, LeTensor *x, LeTensor *y, LeTensor *dJ_dy, GList **dJ_dw);
  // LeShape * (*get_output_shape)(LeLayer *self);
  // const char * (*get_description)(LeLayer *self);
  LE_LAYER_CLASS (klass)->forward_prop = le_activation_layer_forward_prop;
  LE_LAYER_CLASS (klass)->backward_prop = le_activation_layer_backward_prop;
  LE_LAYER_CLASS (klass)->get_output_shape = le_activation_layer_get_output_shape;
  LE_LAYER_CLASS (klass)->get_description = le_activation_layer_get_description;
}

static void
le_activation_layer_init (LeActivationLayer * self)
{

}

#define EPSILON 1e-3f

static LeTensor *
le_tensor_new_softmax_jacobians_stacked(LeTensor *softmax_output)
{  
    LeTensor *self = g_new0 (LeTensor, 1);
    self->element_type = LE_TYPE_F32;
    unsigned num_classes = le_matrix_get_height(softmax_output);
    unsigned num_examples = le_matrix_get_width(softmax_output);
    self->shape = le_shape_new(3, num_examples, num_classes, num_classes);
    self->stride = le_shape_get_size(self->shape, -1);
    self->owns_data = true;
    self->data = g_new0 (gfloat, le_shape_get_elements_count(self->shape));
    
    unsigned num_classes_squared = num_classes * num_classes;
    for (unsigned example = 0; example < num_examples; example++)
    {
        for (unsigned i = 0; i < num_classes; i++)
        {
            gfloat si = le_matrix_at_f32(softmax_output, i, example);
            for (unsigned j = 0; j < num_classes; j++)
            {
                gfloat sj = le_matrix_at_f32(softmax_output, j, example);
                gfloat dJ_daij = (i == j) ? si * (1.0f - si) : -si * sj;
                if (signbit(dJ_daij))
                {
                    if (dJ_daij > -EPSILON)
                        dJ_daij = -EPSILON;
                }
                else
                {
                    if (dJ_daij < EPSILON)
                        dJ_daij = EPSILON;
                }
                ((gfloat *)self->data)[example * num_classes_squared + i * num_classes + j] = dJ_daij;
            }
        }
    }
    
    return self;
}

LeTensor *
le_activation_layer_forward_prop (LeLayer * layer, LeTensor * input)
{
  assert(layer);
  assert(input);
  
  LeActivationLayer *self = LE_ACTIVATION_LAYER (layer);
  LeActivationLayerPrivate *priv = le_activation_layer_get_instance_private (self);
  g_assert_nonnull (priv);
    
  LeTensor *output = le_tensor_new_copy (input);
  switch (priv->activation) {
  case LE_ACTIVATION_SIGMOID:
    /// @note: Sigmoid activation function: g'(x) = 1 / (1 + exp(-x))
    le_tensor_apply_sigmoid(output);
    break;

  case LE_ACTIVATION_TANH:
    /// @note: Hyperbolic tangent activation function: g(x) = tanh(x)
    le_tensor_apply_tanh(output);
    break;

  case LE_ACTIVATION_RELU:
    le_tensor_apply_relu(output);
    break;
      
  case LE_ACTIVATION_SOFTMAX:
    le_matrix_apply_softmax(output);
    break;
      
  case LE_ACTIVATION_LINEAR:
  default:
    /// @note: Linear activation function: g(x) = x
    break;
  }
  return output;
}

LeTensor *
le_activation_layer_backward_prop (LeLayer * layer, LeTensor * cached_input, LeTensor * cached_output, LeTensor * output_gradient, GList ** parameters_gradient)
{
  assert(layer);
  assert(cached_input);
  assert(output_gradient);
  
  LeActivationLayer *self = LE_ACTIVATION_LAYER(layer);
  
  /// @note: Diagonals of Jacobians of activation function at cached_input, stacked.
  /// Rank 2 Tensor. For element-wise activations where a0 depends only from z0.
  LeTensor *activation_primes = NULL;
  /// @note: Jacobians of activation function at cached_input, stacked. Rank 3 Tensor.
  /// For activations where a0 may depend from z0, z1 and other inputs.
  LeTensor *activation_jacobians = NULL;
  /// @note: If both activation_primes and activation_jacobians is NULL, output gradient
  /// will propagade backward unchanged. This is the case for linear activation function.

  LeActivationLayerPrivate *priv = le_activation_layer_get_instance_private (self);
  g_assert_nonnull (priv);

  switch (priv->activation) {
  case LE_ACTIVATION_SIGMOID:
    /// @note: Derivative of sigmoid activation function: g'(x) = g(x)(1 - g(x))
    if (cached_output)
    {
      activation_primes = le_tensor_new_copy(cached_output);
      le_tensor_apply_x_minus_sqr_x(activation_primes);
    }
    else
    {
      activation_primes = le_tensor_new_copy(cached_input);
      le_tensor_apply_sigmoid_prime(activation_primes);
    }
    break;

  case LE_ACTIVATION_TANH:
    /// @note: Derivative of hyperbolic tangent activation function: g'(x) = 1 - g(x)^2
    if (cached_output)
    {
      activation_primes = le_tensor_new_copy(cached_output);
      le_tensor_apply_sqr(activation_primes);
      le_tensor_apply_1_minus(activation_primes);
    }
    else
    {
      activation_primes = le_tensor_new_copy(cached_input);
      le_tensor_apply_tanh(activation_primes);
      le_tensor_apply_sqr(activation_primes);
      le_tensor_apply_1_minus(activation_primes);
    }
    break;

  case LE_ACTIVATION_RELU:
    activation_primes = le_tensor_new_copy(cached_input);
    le_tensor_apply_gt(activation_primes, 0.0f);
    break;
      
  case LE_ACTIVATION_SOFTMAX:
    if (cached_output)
    {
      activation_jacobians = le_tensor_new_softmax_jacobians_stacked(cached_output);
    }
    else
    {
      LeTensor *computed_output = le_tensor_new_copy(cached_input);
      le_matrix_apply_softmax(computed_output);
      activation_jacobians = le_tensor_new_softmax_jacobians_stacked(computed_output);
      le_tensor_unref(computed_output);
    }
    break;
      
  case LE_ACTIVATION_LINEAR:
  default:
    /// @note: Derivative of linear activation function: g'(x) = 1
    break;
  }

  LeTensor *input_gradient = NULL;
  if (activation_primes)
  {
    assert(activation_jacobians == NULL);
    /// @note: Diagonal of Jacobian of activation function.
    /// Non-diagonal partial derivatives will be discarded.
    /// Hadamard is used for chain rule.
    input_gradient = le_tensor_new_copy(output_gradient);
    le_tensor_mul(input_gradient, activation_primes);
    le_tensor_unref(activation_primes);
  } 
  else if (activation_jacobians)
  {
    assert(activation_jacobians->shape->num_dimensions == 3);

    unsigned examples_count = activation_jacobians->shape->sizes[0];
    /// @todo: Optimize, just allocate, do not copy.
    input_gradient = le_tensor_new_copy(output_gradient);
    for (unsigned example = 0; example < examples_count; example++)
    {
      LeTensor *jacobian = le_tensor_pick(activation_jacobians, example);
      LE_INFO("jacobian =\n%s", le_tensor_to_cstr(jacobian));
      unsigned classes_count = le_matrix_get_height(jacobian);
      for (unsigned input = 0; input < classes_count; input++)
      {
        gfloat dJ_dz = 0.0f;
        
        for (unsigned output = 0; output < classes_count; output++)
        {
          gfloat dJ_da = le_matrix_at_f32(output_gradient, output, example);
          gfloat da_dz = le_matrix_at_f32(jacobian, output, input);
          dJ_dz += dJ_da * da_dz;
        }
    
        le_matrix_set_f32(input_gradient, input, example, dJ_dz);
      }
      le_tensor_unref(jacobian);
    }
    le_tensor_unref(activation_jacobians);
  }
  else
  {
    /// @note: Both activation_primes and activation_jacobians is NULL.
    /// It means we have identity activation function with derivative equal to 1.
    /// We will just pass output gradient backward.
    input_gradient = le_tensor_new_copy(output_gradient);
  }
  
  return input_gradient;
}

LeShape *
le_activation_layer_get_output_shape(LeLayer *layer)
{
    /// @note: Check output of previous layer
    return le_shape_new(2, 0, 0);
}

static const char *sigmoid_description = "Sigmoid Activation";
static const char *tanh_description = "Hyperbolic Tangent Activation";
static const char *relu_description = "Rectified Linear Unit";
static const char *softmax_description = "Softmax Activation";
static const char *linear_description = "Identity";

const char *
le_activation_layer_get_description(LeLayer *layer)
{
  assert(layer);
  
  LeActivationLayer *self = LE_ACTIVATION_LAYER(layer);
  LeActivationLayerPrivate *priv = le_activation_layer_get_instance_private (self);
  g_assert_nonnull (priv);

  switch (priv->activation) {
  case LE_ACTIVATION_SIGMOID:
    return sigmoid_description;
    break;

  case LE_ACTIVATION_TANH:
    return tanh_description;
    break;

  case LE_ACTIVATION_RELU:
    return relu_description;
    break;
      
  case LE_ACTIVATION_SOFTMAX:
    return softmax_description;
    break;
      
  case LE_ACTIVATION_LINEAR:
  default:
    return linear_description;
    break;
  }
      
  return linear_description;
}

// static LeActivationLayerClass klass;

// static void
// le_activation_layer_class_ensure_init()
// {
//     static bool initialized = false;
    
//     if (!initialized)
//     {
//         klass.parent.forward_prop = le_activation_layer_forward_prop;
//         klass.parent.backward_prop = le_activation_layer_backward_prop;
//         klass.parent.get_output_shape = le_activation_layer_get_output_shape;
//         klass.parent.get_description = le_activation_layer_get_description;
//         initialized = true;
//     }
// }

LeActivationLayer *
le_activation_layer_new (const char * name, LeActivation activation)
{
  LeActivationLayer *self = g_object_new (le_activation_layer_get_type (), NULL);
  // LeActivationLayer *self = malloc(sizeof(LeActivationLayer));
  // le_layer_construct(LE_LAYER(self), name);
  // le_activation_layer_class_ensure_init();
  // G_OBJECT_GET_CLASS(self) = G_OBJECT_CLASS(&klass);
  LeActivationLayerPrivate *priv = le_activation_layer_get_instance_private (self);
  g_assert_nonnull (priv);
  priv->activation = activation;
  return self;
}

LeActivation
le_activation_layer_get_activation (const LeActivationLayer * self)
{
  g_assert_nonnull (self);
  LeActivationLayerPrivate *priv = le_activation_layer_get_instance_private ((LeActivationLayer *)self);
  g_assert_nonnull (priv);
  return priv->activation;
}
