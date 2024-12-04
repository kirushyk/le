/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEDENSELAYER_H__
#define __LEDENSELAYER_H__

#include <glib.h>
#include "lelayer.h"

G_BEGIN_DECLS

/** @note: Densely-connected (fully-connected) layer */

G_DECLARE_FINAL_TYPE (LeDenseLayer, le_dense_layer, LE, DENSE_LAYER, LeLayer);

#define LE_DENSE_LAYER(a) ((LeDenseLayer *)(a))

LeDenseLayer * le_dense_layer_new (const char *name,
                                   unsigned    inputs,
                                   unsigned    units);

G_END_DECLS

#endif
