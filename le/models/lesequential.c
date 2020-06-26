/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "sequential"

#include "lesequential.h"
#include <assert.h>
#include <stdlib.h>
#include <le/lelog.h>
#include <le/leloss.h>
#include <le/models/layers/leactivationlayer.h>
#include <le/tensors/letensor-imp.h>
#include "lelist.h"
#include <le/tensors/lematrix.h>

struct LeSequential
{
    LeModel parent;
    LeList *layers;
    LeLoss loss;
};

typedef struct LeSequentialClass
{
    LeModelClass parent;
} LeSequentialClass;

static LeSequentialClass klass;

LeTensor *
le_sequential_predict(LeSequential *self, const LeTensor *x);

LeList *
le_sequential_get_gradients(LeSequential *self, const LeTensor *x, const LeTensor *y);

static void
le_sequential_class_ensure_init()
{
    static bool initialized = false;
    
    if (!initialized)
    {
        klass.parent.predict =
            (LeTensor *(*)(LeModel *, const LeTensor *))le_sequential_predict;
        klass.parent.get_gradients =
            (LeList *(*)(LeModel *, const LeTensor *, const LeTensor *))le_sequential_get_gradients;
        initialized = 1;
    }
}

void
le_sequential_construct(LeSequential *self)
{
    le_model_construct((LeModel *)self);
    le_sequential_class_ensure_init();
    ((LeObject *)self)->klass = (LeClass *)&klass;
    
    self->layers = NULL;
    self->loss = LE_LOSS_MSE;
}

LeSequential *
le_sequential_new(void)
{
    LeSequential *self = malloc(sizeof(struct LeSequential));
    le_sequential_construct(self);
    return self;
}

void
le_sequential_add(LeSequential *self, LeLayer *layer)
{
    LE_INFO("Adding New Layer: %s", layer->name);

    self->layers = le_list_append(self->layers, layer);
    LeList *parameters = le_layer_get_parameters(layer);
    
    for (LeList *current = parameters; current != NULL; current = current->next)
    {
        LeTensor *parameter = (LeTensor *)current->data;
        le_model_append_parameter(LE_MODEL(self), parameter);
    }
}

void
le_sequential_set_loss(LeSequential *self, LeLoss loss)
{
    self->loss = loss;
}

/** @note: Used in both _predict and _get_gradients method, 
 * @param inputs if not null is used to cache input of each layer.
 */
static LeTensor *
forward_propagation(LeSequential *self, const LeTensor *x, LeList **inputs)
{
    assert(self);
    assert(x);

    LE_INFO("Forward Propagation");
    LeTensor *signal = le_tensor_new_copy(x);
    
    for (LeList *current = self->layers;
         current != NULL;
         current = current->next)
    {
        LeLayer *current_layer = (LeLayer *)current->data;
        if (inputs)
        {
            *inputs = le_list_append(*inputs, le_tensor_new_copy(signal));
        }
        // LE_INFO("signal =\n%s", le_tensor_to_cstr(signal));
        // LE_INFO("Layer %s Forward", current_layer->name);
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

float 
le_sequential_compute_cost(LeSequential *self, const LeTensor *x, const LeTensor *y)
{
    /// @todo: Take regularization term into account;
    LeTensor *h = forward_propagation(self, x, NULL);
    const float j = le_loss(self->loss, h, y);
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

LeList *
le_sequential_get_gradients(LeSequential *self, const LeTensor *x, const LeTensor *y)
{
    assert(self);
    assert(x);
    assert(y);
        
    /// @note: We cache input of each layer in list of tensors
    /// to ease computation of gradients during backpropagation
    LeList *inputs = NULL;
    LeTensor *signal = forward_propagation(self, x, &inputs);
    // LE_INFO("output =\n%s", le_tensor_to_cstr(signal));
    // LeTensorStats signal_stats = le_tensor_get_stats(signal);
    // LE_INFO("Output stats:\n\tmin: %f\n\tmax: %f\n\tmean: %f\n\tdeviation: %f", signal_stats.min, signal_stats.max, signal_stats.mean, signal_stats.deviation);

    LE_INFO("Back Propagation");
    LeList *current = le_list_last(self->layers);
    inputs = le_list_last(inputs);
    LeActivationLayer *last_layer = NULL;
    LeActivationAndLossBackward actiation_loss_backward = NULL;
    if (current && current->data)
    {
        last_layer = LE_ACTIVATION_LAYER(current->data);
        actiation_loss_backward = activation_loss_backward_fn(last_layer->activation, self->loss);
    }
    if (last_layer && actiation_loss_backward)
    {
        actiation_loss_backward(signal, y);
        current = current->prev;
        inputs = inputs->prev;
    }
    else
    {
        /// @note: Derivative of assumed cost function
        le_apply_loss_derivative(self->loss, signal, y);
        LE_INFO("signal =\n%s", le_tensor_to_cstr(signal));
        // signal_stats = le_tensor_get_stats(signal);
        // LE_INFO("Loss derivative stats:\n\tmin: %f\n\tmax: %f\n\tmean: %f\n\tdeviation: %f", signal_stats.min, signal_stats.max, signal_stats.mean, signal_stats.deviation);
    }

    // LeList *current = NULL;
    LeList *gradients = NULL;
    for (/* current = le_list_last(self->layers), inputs = le_list_last(inputs) */;
         current && inputs;
         current = current->prev, inputs = inputs->prev)
    {
        LeLayer *current_layer = LE_LAYER(current->data);
        LE_INFO("Layer %s Backward", current_layer->name);
        LeList *current_layer_param_gradients = NULL;
        LeTensor *cached_input = LE_TENSOR(inputs->data);
        LeTensor *cached_output = NULL;
        if (inputs->next)
        {
            cached_output = LE_TENSOR(inputs->next->data);
        }
        /// @todo: Use cached output of last layer to speed-up backprop.
        LeTensor *input_gradient = le_layer_backward_prop(current_layer, cached_input, cached_output, signal, &current_layer_param_gradients); 
        le_tensor_free(signal);
        signal = input_gradient;
        LE_INFO("signal =\n%s", le_tensor_to_cstr(signal));
        // LeTensorStats signal_stats = le_tensor_get_stats(signal);
        // LE_INFO("Signal stats:\n\tmin: %f\n\tmax: %f\n\tmean: %f\n\tdeviation: %f", signal_stats.min, signal_stats.max, signal_stats.mean, signal_stats.deviation);
        for (LeList *current_gradient = current_layer_param_gradients;
             current_gradient;
             current_gradient = current_gradient->next)
        {
            LeTensor *gradient = LE_TENSOR(current_gradient->data);
            gradients = le_list_prepend(gradients, gradient);
        }
    }
    
    /// @note: Make sure number of cached inputs equal to number of layers
    assert(current == NULL);
    assert(inputs == NULL);

    le_tensor_free(signal);

    return gradients;
}

LeList *
le_sequential_estimate_gradients(LeSequential *self, const LeTensor *x, const LeTensor *y, float epsilon)
{
    assert(self);
    assert(x);
    assert(y);

    LeList *grad_estimates = NULL;

    for (LeList *params_iterator = LE_MODEL(self)->parameters;
         params_iterator;
         params_iterator = params_iterator->next)
    {
        LeTensor *param = LE_TENSOR(params_iterator->data);
        LeTensor *grad_estimate = le_tensor_new_zeros_like(param);
        unsigned elements_count = le_shape_get_elements_count(param->shape);
        for (unsigned i = 0; i < elements_count; i++)
        {
            const float element = le_tensor_at_f32(param, i);
            le_tensor_set_f32(param, i, element + element * epsilon);
            const float j_plus = le_sequential_compute_cost(self, x, y);
            le_tensor_set_f32(param, i, element - element * epsilon);
            const float j_minus = le_sequential_compute_cost(self, x, y);
            const float element_grad_estimate = (j_plus - j_minus) / (2.0f * element * epsilon);
            le_tensor_set_f32(grad_estimate, i, element_grad_estimate);
            /// @note: We need to restore initial parameter
            le_tensor_set_f32(param, i, element);
        }
        grad_estimates = le_list_append(grad_estimates, grad_estimate);
    }

    return grad_estimates;
}


float
le_sequential_check_gradients(LeSequential *self, const LeTensor *x, const LeTensor *y, float epsilon)
{
    LeList *gradients = le_model_get_gradients(LE_MODEL(self), x, y);
    LeList *gradients_estimations = le_sequential_estimate_gradients(self, x, y, epsilon);
    LeList *gradients_iterator, *gradients_estimations_iterator;
    float average_normalized_distance = 0.0f;
    unsigned parameter_number = 0;
    for (gradients_iterator = gradients, gradients_estimations_iterator = gradients_estimations;
         gradients_iterator && gradients_estimations_iterator;
         gradients_iterator = gradients_iterator->next, gradients_estimations_iterator = gradients_estimations_iterator->next)
    {
        LeTensor *gradient_estimate = (LeTensor *)gradients_estimations_iterator->data;
        LE_INFO("gradient_estimate =\n%s", le_tensor_to_cstr(gradient_estimate));
        LeTensor *gradient = (LeTensor *)gradients_iterator->data;
        LE_INFO("gradient =\n%s", le_tensor_to_cstr(gradient));
        float denominator = le_tensor_l2_f32(gradient) + le_tensor_l2_f32(gradient_estimate);
        float normalized_distance = 1.0f;
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
    le_list_foreach(gradients_estimations, (LeFunction)le_tensor_free);
    le_list_foreach(gradients, (LeFunction)le_tensor_free);
    return average_normalized_distance;
}

void
le_sequential_to_dot(LeSequential *self, const char *filename)
{
    FILE *fout = fopen(filename, "wt");
    
    if (!fout)
        return;

    fprintf(fout, "digraph graphname {\n");
    fprintf(fout, "__cost [shape=record label=\"{J|%s}\"];\n", le_loss_get_desc(self->loss));

    for (LeList *current = self->layers;
         current != NULL; 
         current = current->next)
    {
        LeLayer *current_layer = LE_LAYER(current->data);
        assert(current_layer);
        fprintf(fout, "%s [shape=record label=\"{%s|%s|%d Parameters}\"];\n",
            current_layer->name, current_layer->name,
            le_layer_get_description(current_layer),
            le_layer_get_parameters_count(current_layer));
        const char *next_node = "__cost";
        if (current->next)
        {
            LeLayer *next_layer = LE_LAYER(current->next->data);
            assert(next_layer);

            next_node = next_layer->name;
        }
            
        LeShape *current_laye_output_shape = le_layer_get_output_shape(current_layer);
        fprintf(fout, "%s -> %s [label=\"%s\"];\n", 
            current_layer->name, next_node,
            le_shape_to_cstr(current_laye_output_shape));
        le_shape_free(current_laye_output_shape);

    }

    fprintf(fout, "}\n");
    
    fclose(fout);
}

void
le_sequential_free(LeSequential *self)
{
    free(self);
}
