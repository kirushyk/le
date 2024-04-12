/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "sequential"

#include "lesequential.h"
#include <assert.h>
#include <stdlib.h>
#include <glib.h>
#include <le/lelog.h>
#include <le/leloss.h>
#include <le/models/layers/leactivationlayer.h>
#include <le/tensors/letensor-imp.h>
#include <le/tensors/lematrix.h>

typedef struct _LeSequential
{
    LeModel parent;
} LeSequential;

typedef struct _LeSequentialPrivate
{
    GList *layers;
    LeLoss loss;
} LeSequentialPrivate;

static void le_sequential_class_init (LeSequentialClass * klass);
static void le_sequential_init (LeSequential * self);
G_DEFINE_FINAL_TYPE_WITH_PRIVATE (LeSequential, le_sequential, LE_TYPE_MODEL);

static void
le_sequential_dispose (GObject * object)
{
  G_OBJECT_CLASS (le_sequential_parent_class)->dispose (object);
}

static void
le_sequential_finalize (GObject * object)
{
}

static void
le_sequential_class_init (LeSequentialClass * klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_sequential_dispose;
  G_OBJECT_CLASS (klass)->finalize = le_sequential_finalize;
  LE_MODEL_CLASS (klass)->predict = (LeTensor *(*)(LeModel *, const LeTensor *))le_sequential_predict;
  LE_MODEL_CLASS (klass)->get_gradients = (GList *(*)(LeModel *, const LeTensor *, const LeTensor *))le_sequential_get_gradients;
}

static void
le_sequential_init (LeSequential * self)
{

}

LeTensor *
le_sequential_predict(LeSequential *self, const LeTensor *x);

GList *
le_sequential_get_gradients(LeSequential *self, const LeTensor *x, const LeTensor *y);

LeSequentialClass *
le_sequential_class_ensure_init()
{
    static bool initialized = false;
    static LeSequentialClass klass;
    
    if (!initialized)
    {
        initialized = 1;
    }

    return &klass;
}

// void
// le_sequential_construct(LeSequential *self)
// {
//     le_model_construct(LE_MODEL(self));
//     G_OBJECT_GET_CLASS(self) = G_OBJECT_CLASS(le_sequential_class_ensure_init());
    
//     self->layers = NULL;
//     priv->loss = LE_LOSS_MSE;
// }

LeSequential *
le_sequential_new(void)
{
    LeSequential *self = g_object_new (le_sequential_get_type (), NULL);
    return self;
}

void
le_sequential_add(LeSequential *self, LeLayer *layer)
{
    LE_INFO("Adding New Layer: %s", le_layer_get_name (layer));

    LeSequentialPrivate *priv = le_sequential_get_instance_private (self);
    priv->layers = g_list_append (priv->layers, layer);
    GList *parameters = le_layer_get_parameters(layer);
    
    for (GList *current = parameters; current != NULL; current = current->next)
    {
        LeTensor *parameter = LE_TENSOR(current->data);
        le_model_append_parameter(LE_MODEL(self), parameter);
    }
}

void
le_sequential_set_loss(LeSequential *self, LeLoss loss)
{
    LeSequentialPrivate *priv = le_sequential_get_instance_private (self);
    priv->loss = loss;
}

/** @note: Used in both _predict and _get_gradients method, 
 * @param inputs if not null is used to cache input of each layer.
 */
static LeTensor *
forward_propagation(LeSequential *self, const LeTensor *x, GList **inputs)
{
    assert(self);
    assert(x);
    LeSequentialPrivate *priv = le_sequential_get_instance_private (self);

    LE_INFO("Forward Propagation");
    LeTensor *signal = le_tensor_new_copy(x);
    
    for (GList *current = priv->layers;
         current != NULL;
         current = current->next)
    {
        LeLayer *current_layer = LE_LAYER(current->data);
        if (inputs)
        {
            *inputs = g_list_append(*inputs, le_tensor_new_copy(signal));
        }
        // LE_INFO("signal =\n%s", le_tensor_to_cstr(signal));
        // LE_INFO("Layer %s Forward", le_layer_get_name (current_layer));
        LeTensor *output = le_layer_forward_prop(current_layer, signal);
        le_tensor_free(signal);
        signal = output;
    }

    return signal;
}

LeTensor *
le_sequential_predict(LeSequential *self, const LeTensor *x)
{
    return forward_propagation(self, x, NULL);
}

gfloat 
le_sequential_compute_cost(LeSequential *self, const LeTensor *x, const LeTensor *y)
{
    LeSequentialPrivate *priv = le_sequential_get_instance_private (self);
    /// @todo: Take regularization term into account;
    LeTensor *h = forward_propagation(self, x, NULL);
    const gfloat j = le_loss(priv->loss, h, y);
    le_tensor_free(h);
    return j;
}

typedef void(* LeActivationAndLossBackward)(LeTensor *signal, const LeTensor *labels);

static void
softmax_xent_backward(LeTensor *signal, const LeTensor *labels)
{
    le_tensor_mul(signal, labels);
    le_tensor_sub(signal, labels);
}

static LeActivationAndLossBackward
activation_loss_backward_fn(LeActivation activation, LeLoss loss)
{
    if ((activation == LE_ACTIVATION_SOFTMAX) && (loss == LE_LOSS_CROSS_ENTROPY)) {
        return softmax_xent_backward;
    }
    else if (((activation == LE_ACTIVATION_SIGMOID) && (loss == LE_LOSS_LOGISTIC)) ||
             ((activation == LE_ACTIVATION_LINEAR) && (loss == LE_LOSS_MSE)))
    {
        return le_tensor_sub_tensor;
    }

    return NULL;
}

GList *
le_sequential_get_gradients(LeSequential *self, const LeTensor *x, const LeTensor *y)
{
    assert(self);
    assert(x);
    assert(y);
    LeSequentialPrivate *priv = le_sequential_get_instance_private (self);
        
    /// @note: We cache input of each layer in list of tensors
    /// to ease computation of gradients during backpropagation
    GList *cached_inputs = NULL;
    LeTensor *signal = forward_propagation(self, x, &cached_inputs);
    // LE_INFO("output =\n%s", le_tensor_to_cstr(signal));
    // LeTensorStats signal_stats = le_tensor_get_stats(signal);
    // LE_INFO("Output stats:\n\tmin: %f\n\tmax: %f\n\tmean: %f\n\tdeviation: %f", signal_stats.min, signal_stats.max, signal_stats.mean, signal_stats.deviation);

    LE_INFO("Back Propagation");
    GList *current_layer_iterator = g_list_last(priv->layers);
    GList *cached_inputs_iterator = g_list_last(cached_inputs);
    LeActivationLayer *last_layer = NULL;
    LeActivationAndLossBackward activation_loss_backward = NULL;
    if (current_layer_iterator && current_layer_iterator->data)
    {
        last_layer = LE_ACTIVATION_LAYER(current_layer_iterator->data);
        activation_loss_backward = activation_loss_backward_fn (le_activation_layer_get_activation (last_layer), priv->loss);
    }
    if (last_layer && activation_loss_backward)
    {
        activation_loss_backward(signal, y);
        current_layer_iterator = current_layer_iterator->prev;
        cached_inputs_iterator = cached_inputs_iterator->prev;
    }
    else
    {
        /// @note: Derivative of assumed cost function
        le_apply_loss_derivative(priv->loss, signal, y);
        LE_INFO("signal =\n%s", le_tensor_to_cstr(signal));
        // signal_stats = le_tensor_get_stats(signal);
        // LE_INFO("Loss derivative stats:\n\tmin: %f\n\tmax: %f\n\tmean: %f\n\tdeviation: %f", signal_stats.min, signal_stats.max, signal_stats.mean, signal_stats.deviation);
    }

    // GList *current = NULL;
    GList *gradients = NULL;
    for (/* current = g_list_last(self->layers), inputs = g_list_last(inputs) */;
         current_layer_iterator && cached_inputs_iterator;
         current_layer_iterator = current_layer_iterator->prev, cached_inputs_iterator = cached_inputs_iterator->prev)
    {
        LeLayer *current_layer = LE_LAYER(current_layer_iterator->data);
        LE_INFO("Layer %s Backward", le_layer_get_name (current_layer));
        GList *current_layer_param_gradients = NULL;
        LeTensor *cached_input = LE_TENSOR(cached_inputs_iterator->data);
        LeTensor *cached_output = NULL;
        if (cached_inputs_iterator->next)
        {
            cached_output = LE_TENSOR(cached_inputs_iterator->next->data);
        }
        /// @todo: Use cached output of last layer to speed-up backprop.
        LeTensor *input_gradient = le_layer_backward_prop(current_layer, cached_input, cached_output, signal, &current_layer_param_gradients); 
        le_tensor_free(signal);
        signal = input_gradient;
        LE_INFO("signal =\n%s", le_tensor_to_cstr(signal));
        // LeTensorStats signal_stats = le_tensor_get_stats(signal);
        // LE_INFO("Signal stats:\n\tmin: %f\n\tmax: %f\n\tmean: %f\n\tdeviation: %f", signal_stats.min, signal_stats.max, signal_stats.mean, signal_stats.deviation);
        for (GList *current_gradient = current_layer_param_gradients;
             current_gradient;
             current_gradient = current_gradient->next)
        {
            LeTensor *gradient = LE_TENSOR(current_gradient->data);
            gradients = g_list_prepend(gradients, gradient);
        }
    }
    
    /// @note: Make sure number of cached inputs equal to number of layers
    assert(current_layer_iterator == NULL);
    assert(cached_inputs_iterator == NULL);

    g_list_free_full (cached_inputs, (GDestroyNotify)le_tensor_free);
    le_tensor_free(signal);

    return gradients;
}

GList *
le_sequential_estimate_gradients(LeSequential *self, const LeTensor *x, const LeTensor *y, gfloat epsilon)
{
    assert(self);
    assert(x);
    assert(y);
    assert(epsilon > 0.0f);

    GList *grad_estimates = NULL;

    for (GList *params_iterator = le_model_get_parameters (LE_MODEL (self));
         params_iterator;
         params_iterator = params_iterator->next)
    {
        LeTensor *param = LE_TENSOR(params_iterator->data);
        LeTensor *grad_estimate = le_tensor_new_zeros_like(param);
        unsigned elements_count = le_shape_get_elements_count(param->shape);
        for (unsigned i = 0; i < elements_count; i++)
        {
            const gfloat element = le_tensor_at_f32(param, i);
            le_tensor_set_f32(param, i, element + epsilon);
            const gfloat j_plus = le_sequential_compute_cost(self, x, y);
            le_tensor_set_f32(param, i, element - epsilon);
            const gfloat j_minus = le_sequential_compute_cost(self, x, y);
            const gfloat element_grad_estimate = (j_plus - j_minus) / (2.0f * epsilon);
            le_tensor_set_f32(grad_estimate, i, element_grad_estimate);
            /// @note: We need to restore initial parameter
            le_tensor_set_f32(param, i, element);
        }
        grad_estimates = g_list_append(grad_estimates, grad_estimate);
    }

    return grad_estimates;
}


gfloat
le_sequential_check_gradients(LeSequential *self, const LeTensor *x, const LeTensor *y, gfloat epsilon)
{
    GList *gradients = le_model_get_gradients(LE_MODEL(self), x, y);
    GList *gradients_estimations = le_sequential_estimate_gradients(self, x, y, epsilon);
    GList *gradients_iterator, *gradients_estimations_iterator;
    gfloat average_normalized_distance = 0.0f;
    unsigned parameter_number = 0;
    for (gradients_iterator = gradients, gradients_estimations_iterator = gradients_estimations;
         gradients_iterator && gradients_estimations_iterator;
         gradients_iterator = gradients_iterator->next, gradients_estimations_iterator = gradients_estimations_iterator->next)
    {
        LeTensor *gradient_estimate = LE_TENSOR(gradients_estimations_iterator->data);
        LE_INFO("gradient_estimate =\n%s", le_tensor_to_cstr(gradient_estimate));
        LeTensor *gradient = LE_TENSOR(gradients_iterator->data);
        LE_INFO("gradient =\n%s", le_tensor_to_cstr(gradient));
        gfloat denominator = le_tensor_l2_f32(gradient) + le_tensor_l2_f32(gradient_estimate);
        gfloat normalized_distance = 1.0f;
        if (denominator > 0.0f)
        {
            le_tensor_sub(gradient_estimate, gradient);
            normalized_distance = le_tensor_l2_f32(gradient_estimate);// / denominator;
            LE_INFO("Normalized distance between gradient estimation and actual gradient: %f", normalized_distance);
            if (normalized_distance > epsilon)
            {
                LE_WARNING("Normalized distance too large for parameter #%u: %f", parameter_number, normalized_distance);
            }
        }
        average_normalized_distance += normalized_distance;
        parameter_number++;
    }
    average_normalized_distance /= parameter_number;
    if (gradients_iterator)
    {
        LE_WARNING("Some gradients estimations missing or extra gradients present");
    }
    if (gradients_estimations_iterator)
    {
        LE_ERROR("Some gradients missing or extra gradients estimations present");
    }
    g_list_free_full (gradients_estimations, (GDestroyNotify)le_tensor_free);
    g_list_free_full (gradients, (GDestroyNotify)le_tensor_free);
    return average_normalized_distance;
}

void
le_sequential_to_dot(LeSequential *self, const char *filename)
{
    LeSequentialPrivate *priv = le_sequential_get_instance_private (self);
    FILE *fout = fopen(filename, "wt");
    
    if (!fout)
        return;

    fprintf(fout, "digraph graphname {\n");
    fprintf(fout, "__cost [shape=record label=\"{J|%s}\"];\n", le_loss_get_desc(priv->loss));

    for (GList *current = priv->layers;
         current != NULL; 
         current = current->next)
    {
        LeLayer *current_layer = LE_LAYER(current->data);
        assert(current_layer);
        fprintf(fout, "%s [shape=record label=\"{%s|%s|%d Parameters}\"];\n",
            le_layer_get_name (current_layer), le_layer_get_name (current_layer),
            le_layer_get_description(current_layer),
            le_layer_get_parameters_count(current_layer));
        const char *next_node = "__cost";
        if (current->next)
        {
            LeLayer *next_layer = LE_LAYER(current->next->data);
            assert(next_layer);

            next_node = le_layer_get_name (next_layer);
        }
            
        LeShape *current_laye_output_shape = le_layer_get_output_shape(current_layer);
        fprintf(fout, "%s -> %s [label=\"%s\"];\n", 
            le_layer_get_name (current_layer), next_node,
            le_shape_to_cstr(current_laye_output_shape));
        le_shape_free(current_laye_output_shape);

    }

    fprintf(fout, "}\n");
    
    fclose(fout);
}

void
le_sequential_free(LeSequential *self)
{
    g_free (self);
}
