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

LeTensor *
le_dense_layer_backward_prop(LeLayer *layer, LeTensor *cached_input, LeTensor *output_gradient, LeList **parameters_gradient)
{
    assert(layer);
    assert(output_gradient);
    
    LeDenseLayer *self = LE_DENSE_LAYER(layer);

    assert(self->w);

    LeTensor *input_gradient = le_matrix_new_product_full(self->w, true, output_gradient, false);

    if (parameters_gradient)
    {
        assert(cached_input);

        LeTensor *h = le_tensor_new_copy(output_gradient);
        unsigned examples_count = le_matrix_get_width(h);
        le_tensor_multiply_by_scalar(h, 1.0f / examples_count);
        LeTensor *dw = le_matrix_new_product_full(h, false, cached_input, true);
        LeTensor *db = le_matrix_new_sum(h, 1);
        le_tensor_free(h);
        *parameters_gradient = le_list_append(*parameters_gradient, db);
        *parameters_gradient = le_list_append(*parameters_gradient, dw);
    }

    return input_gradient;
}

LeShape *
le_dense_layer_get_output_shape(LeLayer *layer)
{
    LeDenseLayer *self = LE_DENSE_LAYER(layer);
    
    return le_shape_new(2, le_matrix_get_height(self->b), 0);
}

static LeDenseLayerClass klass;

static void
le_dense_layer_class_ensure_init()
{
    static bool initialized = false;
    
    if (!initialized)
    {
        klass.parent.forward_prop = le_dense_layer_forward_prop;
        klass.parent.backward_prop = le_dense_layer_backward_prop;
        klass.parent.get_output_shape = le_dense_layer_get_output_shape;
        initialized = true;
    }
}

LeDenseLayer *
le_dense_layer_new(const char *name, unsigned inputs, unsigned units)
{
    LeDenseLayer *self = malloc(sizeof(LeDenseLayer));
    le_layer_construct(LE_LAYER(self), name);
    le_dense_layer_class_ensure_init();
    LE_OBJECT_GET_CLASS(self) = LE_CLASS(&klass);
    self->w = le_matrix_new_rand(units, inputs);
    /// @todo: Optimize
    le_tensor_multiply_by_scalar(self->w, 2.0f);
    le_tensor_subtract_scalar(self->w, 1.0f);
    self->b = le_matrix_new_zeros(units, 1);
    le_layer_append_parameter(LE_LAYER(self), self->w);
    le_layer_append_parameter(LE_LAYER(self), self->b);
    return self;
}
