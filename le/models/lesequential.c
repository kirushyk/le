/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "sequential"

#include "lesequential.h"
#include <le/lelog.h>
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

LeSequentialClass le_sequential_class;

LeTensor *
le_sequential_predict(LeSequential *self, LeTensor *x);

LeList *
le_sequential_get_gradients(LeSequential *self, LeTensor *x, LeTensor *y);

static void
le_sequential_class_ensure_init()
{
    static int le_sequential_class_initialized = 0;
    
    if (!le_sequential_class_initialized)
    {
        le_sequential_class.parent.predict =
        (LeTensor *(*)(LeModel *, LeTensor *))le_sequential_predict;
        le_sequential_class.parent.get_gradients =
            (LeList *(*)(LeModel *, LeTensor *, LeTensor *))le_sequential_get_gradients;
        le_sequential_class_initialized = 1;
    }
}

void
le_sequential_construct(LeSequential *self)
{
    le_model_construct((LeModel *)self);
    le_sequential_class_ensure_init();
    ((LeObject *)self)->klass = (LeClass *)&le_sequential_class;
    
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
    self->layers = le_list_append(self->layers, layer);
    LeList *parameters = le_layer_get_parameters(layer);
    
    for (LeList *current = parameters; current != NULL; current = current->next)
    {
        LeTensor *parameter = (LeTensor *)current->data;
        le_model_append_parameter(LE_MODEL(self), parameter);
    }
}

LeTensor *
le_sequential_predict(LeSequential *self, LeTensor *x)
{
    assert(self);
    assert(x);
    
    LeTensor *signal = le_tensor_new_copy(x);
    for (LeList *current = self->layers; current != NULL; current = current->next)
    {
        LeLayer *current_layer = (LeLayer *)current->data;
        printf("signal =\n");
        le_tensor_print(signal, stdout);
        LeTensor *output = le_layer_forward_prop(current_layer, signal);
        le_tensor_free(signal);
        signal = output;
    }
    return signal;
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
    
    LeTensor *signal = le_tensor_new_copy(x);
    
    LeList *current;

    LE_INFO("Forward Propagation");
    
    for (current = self->layers;
         current != NULL; 
         current = current->next)
    {
        LeLayer *current_layer = (LeLayer *)current->data;
        inputs = le_list_append(inputs, le_tensor_new_copy(signal));
        LE_INFO("signal =");
        le_tensor_print(signal, stdout);
        LE_INFO("Layer: %s", current_layer->name);
        LeTensor *output = le_layer_forward_prop(current_layer, signal);
        le_tensor_free(signal);
        signal = output;
    }

    LE_INFO("output =");
    le_tensor_print(signal, stdout);
    
    LE_INFO("Back Propagation");
    
    /// @note: Derivative of assumed cost function
    le_tensor_subtract(signal, y);

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
        LE_INFO("signal =");
        le_tensor_print(signal, stdout);
        for (LeList *current_gradient = current_layer_param_gradients;
             current_gradient != NULL;
             current_gradient = current_gradient->next)
        {
            LeTensor *gradient = LE_TENSOR(current_gradient->data);
            gradients = le_list_prepend(gradients, gradient);
        }
    }
    
    assert(current == NULL);
    assert(inputs == NULL);

    le_tensor_free(signal);

    return gradients;
}

void
le_sequential_free(LeSequential *self)
{
    free(self);
}
