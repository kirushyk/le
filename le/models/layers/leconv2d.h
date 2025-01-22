/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_MODELS_LAYERS_CONV2D_H__
#define __LE_MODELS_LAYERS_CONV2D_H__

#include "lelayer.h"
#include <glib.h>

G_BEGIN_DECLS

typedef struct LeConv2D
{
  LeLayer parent;

  unsigned int padding;
  unsigned int stride;

  LeTensor *w;
  LeTensor *b;
} LeConv2D;

#define LE_CONV2D(a) ((LeConv2D *)(a))

LeConv2D * le_conv2d_new (const char *name,
                          unsigned    filter_size,
                          unsigned    num_channels,
                          unsigned    num_filters,
                          unsigned    padding,
                          unsigned    stride);

G_END_DECLS

#endif
