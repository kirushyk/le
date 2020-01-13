/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_LAYER_H__
#define __LE_LAYER_H__

#include "../leobject.h"
#include "../letensor.h"

/** @note: Abstract Layer class */

typedef struct LeLayer
{
    LeObject parent;
} LeLayer;

#define LE_LAYER(a) ((LeLayer *)(a))

typedef struct LeLayerClass
{
    LeClass parent;
    
    LeTensor * (*forward_prop)(LeLayer *model, LeTensor *x);
} LeLayerClass;

#define LE_LAYER_CLASS(a) ((LeLayerClass *)(a))

LeTensor * le_layer_forward_prop (LeLayer  *layer,
                                  LeTensor *input);

/** @note: Densely-connected (fully-connected) layer */

typedef struct LeDenseLayer
{
    LeLayer parent;
    
    LeTensor *w;
    LeTensor *b;
} LeDenseLayer;

#define LE_DENSE_LAYER(a) ((LeDenseLayer *)(a))

LeDenseLayer * le_dense_layer_new (unsigned inputs,
                                   unsigned units);

typedef enum LeActivation
{
    LE_ACTIVATION_LINEAR,
    LE_ACTIVATION_TANH,
    LE_ACTIVATION_SOFTMAX
} LeActivation;

typedef struct LeActivationLayer
{
    LeLayer parent;
    
    LeActivation activation;
} LeActivationLayer;

#define LE_ACTIVATION_LAYER(a) ((LeActivationLayer *)(a))

LeActivationLayer * le_activation_layer_new (LeActivation activation);

#endif
