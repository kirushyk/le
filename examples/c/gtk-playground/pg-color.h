/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __GTK_PLAYGROUND_PG_COLOR_H__
#define __GTK_PLAYGROUND_PG_COLOR_H__

#include <glib.h>

typedef struct BGRA32
{
  guint8 b;
  guint8 g;
  guint8 r;
  guint8 a;
} BGRA32;

BGRA32 color_for_logistic (gfloat scalar);

#endif
