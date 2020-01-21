/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_LAYER_H__
#define __LE_LAYER_H__

#include <le/leobject.h>
#include <le/letensor.h>
#include <le/lelist.h>

/** @note: Abstract Layer class */

typedef struct LeLayer
{
    LeObject parent;
    
    LeList     *parameters;
    const char *name;
} LeLayer;

#define LE_LAYER(a) ((LeLayer *)(a))

typedef struct LeLayerClass
{
    LeClass parent;
    
    LeTensor * (*forward_prop)(LeLayer *model, LeTensor *x);
} LeLayerClass;

#define LE_LAYER_CLASS(a) ((LeLayerClass *)(a))

void       le_layer_construct        (LeLayer    *layer,
                                      const char *name);

LeTensor * le_layer_forward_prop     (LeLayer    *layer,
                                      LeTensor   *input);

LeList *   le_layer_get_parameters   (LeLayer    *layer);

void       le_layer_append_parameter (LeLayer    *layer,
                                      LeTensor   *parameter);

LeList *   le_layer_get_gradients    (LeLayer    *layer);

#endif
