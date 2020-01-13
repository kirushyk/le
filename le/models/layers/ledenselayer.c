/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "ledenselayer.h"
#include <assert.h>
#include <stdlib.h>
#include <le/lematrix.h>

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
    le_layer_construct(LE_LAYER(self));
    le_dense_layer_class_ensure_init();
    LE_OBJECT_GET_CLASS(self) = LE_CLASS(&le_dense_layer_class);
    self->w = le_matrix_new_rand(units, inputs);
    self->b = le_matrix_new_rand(units, 1);
    le_layer_append_parameter(LE_LAYER(self), self->w);
    le_layer_append_parameter(LE_LAYER(self), self->b);
    return self;
}
