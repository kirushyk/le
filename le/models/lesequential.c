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
    
//    unsigned examples_count = le_matrix_get_width(y);
    
    /// @note: We cache output of each layer in list of tensors
    /// to ease computation of gradients during backpropagation
    LeList *outputs = NULL;
    
    LeTensor *signal = le_tensor_new_copy(x);
    
    LeList *current;

    LE_INFO("Forward Propagation");
    
    for (current = self->layers;
         current != NULL; 
         current = current->next)
    {
        LeLayer *current_layer = (LeLayer *)current->data;
        LE_INFO("signal =");
        le_tensor_print(signal, stdout);
        LE_INFO("Layer: %s", current_layer->name);
        LeTensor *output = le_layer_forward_prop(current_layer, signal);
        le_tensor_free(signal);
        signal = output;
        
        outputs = le_list_append(outputs, le_tensor_new_copy(output));
    }

    LE_INFO("Back Propagation");
    
    /// @note: Derivative of assumed cost function
    le_tensor_subtract(signal, y);

//    LeTensor *h = le_sequential_predict(self, x);
//    le_tensor_subtract(h, y);
//    le_tensor_multiply_by_scalar(h, 1.0 / examples_count);
//    LeTensor *dw = le_matrix_new_product_full(h, false, x, true);
//    LeTensor *db = le_matrix_new_sum(h, 1);
//    le_tensor_free(h);

    LeList *gradients = NULL;
    for (current = le_list_last(self->layers), outputs = le_list_last(outputs);
         current && outputs;
         current = current->prev, outputs = outputs->prev)
    {
        LeLayer *current_layer = LE_LAYER(current->data);
        LE_INFO("Layer: %s", current_layer->name);
        LeList *layer_gradients = le_layer_get_gradients(current_layer);
        for (LeList *current_gradient = layer_gradients;
             current_gradient != NULL;
             current_gradient = current_gradient->next)
        {
            LeTensor *gradient = LE_TENSOR(current_gradient->data);
            gradients = le_list_append(gradients, gradient);
        }
    }
    
    assert(current == NULL);
    assert(outputs == NULL);

    le_tensor_free(signal);

    return gradients;
}

void
le_sequential_free(LeSequential *self)
{
    free(self);
}
