/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leconv2d.h"
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <le/lematrix.h>

typedef struct LeConv2DClass
{
    LeLayerClass parent;
    
} LeConv2DClass;

static LeConv2DClass klass;

LeTensor *
le_conv2d_forward_prop(LeLayer *layer, LeTensor *input)
{
    return NULL;
}

LeTensor *
le_conv2d_backward_prop(LeLayer *layer, LeTensor *cached_input, LeTensor *cached_output,
                        LeTensor *output_gradient, LeList **parameters_gradient)
{
    return NULL;
}

const char *
le_conv2d_get_description(LeLayer *self)
{
    static const char *description = "2D Convolutional Layer";
    return description;
}

static void
le_conv2d_class_ensure_init()
{
    static bool initialized = false;
    
    if (!initialized)
    {
        klass.parent.forward_prop = le_conv2d_forward_prop;
        klass.parent.backward_prop = le_conv2d_backward_prop;
        klass.parent.get_output_shape = NULL;
        klass.parent.get_description = le_conv2d_get_description;
        initialized = true;
    }
}

LeConv2D *
le_conv2d_new(const char *name, unsigned filter_size, unsigned num_channels, 
              unsigned num_filters, unsigned padding, unsigned stride)
{
    LeConv2D *self = malloc(sizeof(LeConv2D));
    le_layer_construct(LE_LAYER(self), name);
    le_conv2d_class_ensure_init();
    LE_OBJECT_GET_CLASS(self) = LE_CLASS(&klass);
    LeShape *weights_shape = le_shape_new(4, filter_size, filter_size, num_channels, num_filters);
    self->w = le_tensor_new_rand_f32(weights_shape);
    self->b = le_matrix_new_zeros(LE_TYPE_FLOAT32, num_filters, 1);
    le_layer_append_parameter(LE_LAYER(self), self->w);
    le_layer_append_parameter(LE_LAYER(self), self->b);
    return self;
}
