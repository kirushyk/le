/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lelayer.h"
#include <assert.h>
#include <stdlib.h>
#include "../lematrix.h"

LeTensor *
le_layer_forward_prop(LeLayer *layer, LeTensor *input)
{
    assert(layer);
    LeLayerClass *klass = (LeLayerClass *)LE_OBJECT_CLASS(layer);
    assert(klass);
    assert(klass->forward_prop);

    return klass->forward_prop(layer, input);
}

typedef struct LeDenseLayerClass
{
    LeLayerClass parent;
    
} LeDenseLayerClass;

LeTensor *
le_dense_layer_forward_prop(LeLayer *layer, LeTensor *input)
{
    assert(layer);
    assert(input);
    
    LeDenseLayer *self = LE_DENSE_LAYER(layer);
    
    assert(self->w);

    LeTensor *output = le_matrix_new_product(self->w, input);
    
    if (self->b)
    {
        le_matrix_add(output, self->b);
    }
    
    return output;
}

static LeDenseLayerClass le_dense_layer_class;

static void
le_dense_layer_class_ensure_init()
{
    static bool le_dense_layer_class_initialized = false;
    
    if (!le_dense_layer_class_initialized)
    {
        le_dense_layer_class.parent.forward_prop = le_dense_layer_forward_prop;
        le_dense_layer_class_initialized = true;
    }
}

LeDenseLayer *
le_dense_layer_new(unsigned inputs, unsigned units)
{
    LeDenseLayer *self = malloc(sizeof(LeDenseLayer));
    le_dense_layer_class_ensure_init();
    LE_OBJECT_CLASS(self) = LE_CLASS(&le_dense_layer_class);
    self->w = le_matrix_new_rand(units, inputs);
    self->b = le_matrix_new_rand(units, 1);
    return self;
}

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
    case LE_ACTIVATION_TANH:
        le_tensor_apply_tanh(output);
        break;
        
    case LE_ACTIVATION_SOFTMAX:
        {
            /// @todo: Optimize
            float scalar = 1.0f / le_matrix_get_height(output);
            le_tensor_apply_sigmoid(output);
            le_tensor_multiply_by_scalar(output, scalar);
        }
        break;
        
    case LE_ACTIVATION_LINEAR:
    default:
        break;
    }
    return output;
}

static LeActivationLayerClass le_activation_layer_class;

static void
le_activation_layer_class_ensure_init()
{
    static bool le_activation_layer_class_initialized = false;
    
    if (!le_activation_layer_class_initialized)
    {
        le_activation_layer_class.parent.forward_prop = le_activation_layer_forward_prop;
        le_activation_layer_class_initialized = true;
    }
}

LeActivationLayer *
le_activation_layer_new(LeActivation activation)
{
    LeActivationLayer *self = malloc(sizeof(LeActivationLayer));
    le_activation_layer_class_ensure_init();
    LE_OBJECT_CLASS(self) = LE_CLASS(&le_activation_layer_class);
    self->activation = activation;
    return self;
}
