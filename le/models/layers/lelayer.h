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
    
    LeTensor * (*forward_prop)(LeLayer *self, LeTensor *x);
    LeTensor * (*backward_prop)(LeLayer *self, LeTensor *x, LeTensor *y, LeTensor *dJ_dy, LeList **dJ_dw);
    LeShape * (*get_output_shape)(LeLayer *self);
    const char * (*get_description)(LeLayer *self);
} LeLayerClass;

#define LE_LAYER_CLASS(a) ((LeLayerClass *)(a))
#define LE_LAYER_GET_CLASS(a) LE_LAYER_CLASS(LE_OBJECT_GET_CLASS(a))

void         le_layer_construct            (LeLayer     *layer,
                                            const char  *name);

LeTensor *   le_layer_forward_prop         (LeLayer     *layer,
                                            LeTensor    *input);

LeList *     le_layer_get_parameters       (LeLayer     *layer);

unsigned     le_layer_get_parameters_count (LeLayer     *layer);

void         le_layer_append_parameter     (LeLayer     *layer,
                                            LeTensor    *parameter);

LeTensor *   le_layer_backward_prop        (LeLayer     *layer,
                                            LeTensor    *cached_input,
                                            LeTensor    *cached_output,
                                            LeTensor    *output_gradient,
                                            LeList     **parameters_gradient);

LeShape *    le_layer_get_output_shape     (LeLayer     *layer);

const char * le_layer_get_description      (LeLayer     *layer);

#endif
