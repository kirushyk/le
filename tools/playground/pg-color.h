/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef _LE_PG_COLOR_
#   define _LE_PG_COLOR_
#   include <glib.h>

typedef struct ARGB32
{
    guint8 b, g, r, a;
} ARGB32;

ARGB32 color_for_logistic(float scalar);

#endif
