/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LECONV2D_H__
#define __LECONV2D_H__

#include "lelayer.h"
#include <le/lemacros.h>

LE_BEGIN_DECLS

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

LE_END_DECLS

#endif
