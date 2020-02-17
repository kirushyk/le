/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "sequential"

#include "lesequential.h"
#include <le/lelog.h>
#include <le/leloss.h>
#include <assert.h>
#include <stdlib.h>
#include "lelist.h"
#include "lematrix.h"

struct LeSequential
{
    LeModel parent;
    LeList *layers;
};

typedef struct LeSequentialClass
{
    LeModelClass parent;
} LeSequentialClass;

static LeSequentialClass klass;

LeTensor *
le_sequential_predict(LeSequential *self, LeTensor *x);

LeList *
le_sequential_get_gradients(LeSequential *self, LeTensor *x, LeTensor *y);

static void
le_sequential_class_ensure_init()
{
    static bool initialized = false;
    
    if (!initialized)
    {
        klass.parent.predict =
        (LeTensor *(*)(LeModel *, LeTensor *))le_sequential_predict;
        klass.parent.get_gradients =
            (LeList *(*)(LeModel *, LeTensor *, LeTensor *))le_sequential_get_gradients;
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

/** @note: Used in both _predict and _get_gradients method, 
 * @param inputs if not null is used to cache input of each layer.
 */
static LeTensor *
forward_propagation(LeSequential *self, LeTensor *x, LeList **inputs)
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
        LE_INFO("signal =\n%s", le_tensor_to_cstr(signal));
        LE_INFO("Layer: %s", current_layer->name);
        LeTensor *output = le_layer_forward_prop(current_layer, signal);
        le_tensor_free(signal);
        signal = output;
    }

    return signal;
}

LeTensor *
le_sequential_predict(LeSequential *self, LeTensor *x)
{
    return forward_propagation(self, x, NULL);
}

LeList *
le_sequential_get_gradients(LeSequential *self, LeTensor *x, LeTensor *y)
{
    assert(self);
    assert(x);
    assert(y);
        
    /// @note: We cache input of each layer in list of tensors
    /// to ease computation of gradients during backpropagation
    LeList *inputs = NULL;
    LeTensor *signal = forward_propagation(self, x, &inputs);
    LE_INFO("output =\n%s", le_tensor_to_cstr(signal));

    LE_INFO("Back Propagation");
    /// @note: Derivative of assumed cost function
    le_apply_logistic_loss_derivative(signal, y);
    LE_INFO("signal =\n%s", le_tensor_to_cstr(signal));

    LeList *current = NULL;
    LeList *gradients = NULL;
    for (current = le_list_last(self->layers), inputs = le_list_last(inputs);
         current && inputs;
         current = current->prev, inputs = inputs->prev)
    {
        LeLayer *current_layer = LE_LAYER(current->data);
        LE_INFO("Layer: %s", current_layer->name);
        LeList *current_layer_param_gradients = NULL;
        LeTensor *cached_input = LE_TENSOR(inputs->data);
        LeTensor *input_gradient = le_layer_backward_prop(current_layer, cached_input, signal, &current_layer_param_gradients); 
        le_tensor_free(signal);
        signal = input_gradient;
        LE_INFO("signal =\n%s", le_tensor_to_cstr(signal));
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

void
le_sequential_to_dot(LeSequential *self, const char *filename)
{
    FILE *fout = fopen(filename, "wt");
    
    if (!fout)
        return;

    fprintf(fout, "digraph graphname {\n");
    fprintf(fout, "__cost [shape=record label=\"{J|Cross-entropy cost}\"];\n");

    for (LeList *current = self->layers;
         current != NULL; 
         current = current->next)
    {
        LeLayer *current_layer = LE_LAYER(current->data);
        assert(current_layer);
        fprintf(fout, "%s [shape=record label=\"{%s|%s}\"];\n",
            current_layer->name, current_layer->name,
            le_layer_get_description(current_layer));
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
