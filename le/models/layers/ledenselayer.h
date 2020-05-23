/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEDENSELAYER_H__
#define __LEDENSELAYER_H__

#include <le/lemacros.h>
#include "lelayer.h"

LE_BEGIN_DECLS

/** @note: Densely-connected (fully-connected) layer */

typedef struct LeDenseLayer
{
    LeLayer parent;
    
    LeTensor *w;
    LeTensor *b;
} LeDenseLayer;

#define LE_DENSE_LAYER(a) ((LeDenseLayer *)(a))

LeDenseLayer * le_dense_layer_new (const char *name,
                                   unsigned    inputs,
                                   unsigned    units);

LE_END_DECLS

#endif
