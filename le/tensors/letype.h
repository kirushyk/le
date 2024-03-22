/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LETYPE_H__
#define __LETYPE_H__

#include <stddef.h>
#include <stdint.h>
#include <glib.h>

G_BEGIN_DECLS

#define F16_0 (guint16)0
#define F16_1 (guint16)15360
typedef guint16 lehalf;

typedef enum LeType
{
    LE_TYPE_VOID,
    LE_TYPE_INT8,
    LE_TYPE_UINT8,
    LE_TYPE_INT16,
    LE_TYPE_UINT16,
    LE_TYPE_INT32,
    LE_TYPE_UINT32,
    LE_TYPE_FLOAT16,
    LE_TYPE_FLOAT32,
    LE_TYPE_FLOAT64,

    LE_TYPE_COUNT
} LeType;

size_t             le_type_size                            (LeType                  type);

const char *       le_type_name                            (LeType                  type);

G_END_DECLS

#endif
