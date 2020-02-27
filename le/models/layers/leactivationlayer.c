/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leactivationlayer.h"
#include <assert.h>
#include <stdlib.h>
#include <le/lematrix.h>

typedef struct LeActivationLayerClass
{
    LeLayerClass parent;
    
} LeActivationLayerClass;

LeTensor *
le_activation_layer_forward_prop(LeLayer *layer, LeTensor *input)
{
    assert(layer);
    assert(input);
    
    LeActivationLayer *self = LE_ACTIVATION_LAYER(layer);
    
    LeTensor *output = le_tensor_new_copy(input);
    switch (self->activation) {
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
le_activation_layer_backward_prop(LeLayer *layer, LeTensor *cached_input, LeTensor *cached_output, LeTensor *output_gradient, LeList **parameters_gradient)
{
    assert(layer);
    assert(output_gradient);
    
    LeActivationLayer *self = LE_ACTIVATION_LAYER(layer);
    
    /// @note: Diagonals of Jacobians of activation function at cached_input, stacked
    LeTensor *activation_primes = NULL;

    switch (self->activation) {
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
        le_tensor_apply_greater_than(activation_primes, 0.0f);
        break;
        
    case LE_ACTIVATION_SOFTMAX:
        /// @note: Diagonal of Jacobian of softmax function.
        /// Non-diagonal partial derivatives will be discarded.
        /// Hadamard product will be used for chain rule.
        if (cached_output)
        {
            activation_primes = le_tensor_new_copy(cached_output);
            le_tensor_apply_x_minus_sqr_x(activation_primes);
        }
        else
        {
            activation_primes = le_tensor_new_copy(cached_input);
            le_matrix_apply_softmax_prime(activation_primes);
        }
        break;
        
    case LE_ACTIVATION_LINEAR:
        /// @note: Derivative of linear activation function: g'(x) = 1
    default:
        /// @note: NULL activation_primes will be treated like all-ones array
        activation_primes = NULL;
        break;
    }

    LeTensor *input_gradient = le_tensor_new_copy(output_gradient);
    if (activation_primes)
    {
        le_tensor_multiply_elementwise(input_gradient, activation_primes);
        le_tensor_free(activation_primes);
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

    switch (self->activation) {
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

static LeActivationLayerClass klass;

static void
le_activation_layer_class_ensure_init()
{
    static bool initialized = false;
    
    if (!initialized)
    {
        klass.parent.forward_prop = le_activation_layer_forward_prop;
        klass.parent.backward_prop = le_activation_layer_backward_prop;
        klass.parent.get_output_shape = le_activation_layer_get_output_shape;
        klass.parent.get_description = le_activation_layer_get_description;
        initialized = true;
    }
}

LeActivationLayer *
le_activation_layer_new(const char *name, LeActivation activation)
{
    LeActivationLayer *self = malloc(sizeof(LeActivationLayer));
    le_layer_construct(LE_LAYER(self), name);
    le_activation_layer_class_ensure_init();
    LE_OBJECT_GET_CLASS(self) = LE_CLASS(&klass);
    self->activation = activation;
    return self;
}
