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
le_activation_layer_new(const char *name, LeActivation activation)
{
    LeActivationLayer *self = malloc(sizeof(LeActivationLayer));
    le_layer_construct(LE_LAYER(self), name);
    le_activation_layer_class_ensure_init();
    LE_OBJECT_GET_CLASS(self) = LE_CLASS(&le_activation_layer_class);
    self->activation = activation;
    return self;
}
