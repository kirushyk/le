/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lelayer.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <le/lemem.h>
#include <le/tensors/lematrix.h>
#include <le/tensors/letensor-imp.h>

void
le_layer_construct(LeLayer *self, const char *name)
{
    assert(self);
    
    self->parameters = NULL;
    self->name = le_strdup(name);
}

LeTensor *
le_layer_forward_prop(LeLayer *self, LeTensor *input)
{
    assert(self);
    LeLayerClass *klass = LE_LAYER_GET_CLASS(self);
    assert(klass);
    assert(klass->forward_prop);

    return klass->forward_prop(self, input);
}

LeList *
le_layer_get_parameters(LeLayer *self)
{
    assert(self);
    
    return self->parameters;
}

unsigned     
le_layer_get_parameters_count(LeLayer *layer)
{
    unsigned count = 0;
    for (LeList *current = layer->parameters; current != NULL; current = current->next)
    {
        count += le_shape_get_elements_count(LE_TENSOR(current->data)->shape);
    }
    return count;
}

void
le_layer_append_parameter(LeLayer *self, LeTensor *parameter)
{
    assert(self);
    assert(parameter);
    
    self->parameters = le_list_append(self->parameters, parameter);
}

LeTensor * 
le_layer_backward_prop(LeLayer *self, LeTensor *cached_input, LeTensor *cached_output, LeTensor *output_gradient, LeList **parameters_gradient)
{
    assert(self);
    LeLayerClass *klass = LE_LAYER_GET_CLASS(self);
    assert(klass);
    assert(klass->backward_prop);
    assert(output_gradient);
    
    return klass->backward_prop(self, cached_input, cached_output, output_gradient, parameters_gradient);
}

LeShape *
le_layer_get_output_shape(LeLayer *self)
{
    assert(self);
    LeLayerClass *klass = LE_LAYER_GET_CLASS(self);
    assert(klass);
    assert(klass->get_output_shape);

    return klass->get_output_shape(self);
}

const char *
le_layer_get_description(LeLayer *self)
{
    assert(self);
    LeLayerClass *klass = LE_LAYER_GET_CLASS(self);
    assert(klass);
    assert(klass->get_description);

    return klass->get_description(self);
}
