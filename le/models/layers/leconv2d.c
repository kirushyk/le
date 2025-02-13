/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leconv2d.h"
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <le/tensors/lematrix.h>
#include <le/tensors/letensor-imp.h>

typedef struct LeConv2DClass
{
    LeLayerClass parent;
    
} LeConv2DClass;

static LeConv2DClass klass;

LeTensor *
le_conv2d_forward_prop(LeLayer *layer, LeTensor *input)
{
    assert(layer);
    assert(input);

    LeConv2D *self = LE_CONV2D(layer);

    assert(self->w);
    assert(self->w->shape);
    assert(self->w->shape->num_dimensions == 4);

    unsigned int filter_size_h = self->w->shape->sizes[0];
    unsigned int filter_size_w = self->w->shape->sizes[1];
    unsigned int num_channels = self->w->shape->sizes[2];
    unsigned int num_filters = self->w->shape->sizes[3];

    assert(input->shape);
    assert(input->shape->num_dimensions == 4);

    unsigned int batch_size = input->shape->sizes[0];
    unsigned int input_h = input->shape->sizes[1];
    unsigned int input_w = input->shape->sizes[2];
    unsigned int input_channels_count = input->shape->sizes[3];
    assert(num_channels == input_channels_count);

    unsigned int output_h = (input_h + 2 * self->padding - filter_size_h) / self->stride + 1;
    unsigned int output_w = (input_w + 2 * self->padding - filter_size_w) / self->stride + 1;

    LeShape *output_shape = le_shape_new(4, batch_size, output_h, output_w, num_filters);
    LeTensor *output = le_tensor_new_rand_f32(output_shape);

    return output;
}

LeTensor *
le_conv2d_backward_prop(LeLayer *layer, LeTensor *cached_input, LeTensor *cached_output,
                        LeTensor *output_gradient, GList **parameters_gradient)
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
    G_OBJECT_GET_CLASS(self) = G_OBJECT_CLASS(&klass);
    self->padding = padding;
    self->stride = stride;
    LeShape *weights_shape = le_shape_new(4, filter_size, filter_size, num_channels, num_filters);
    self->w = le_tensor_new_rand_f32(weights_shape);
    LeShape *biases_shape = le_shape_new(4, 1, 1, 1, num_filters);
    self->b = le_tensor_new_rand_f32(biases_shape);
    le_layer_append_parameter(LE_LAYER(self), self->w);
    le_layer_append_parameter(LE_LAYER(self), self->b);
    return self;
}
