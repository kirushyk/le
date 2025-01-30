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
    LE_TYPE_I8,
    LE_TYPE_U8,
    LE_TYPE_I16,
    LE_TYPE_U16,
    LE_TYPE_I32,
    LE_TYPE_U32,
    LE_TYPE_F16,
    LE_TYPE_F32,
    LE_TYPE_F64,

    LE_TYPE_COUNT
} LeType;

gsize             le_type_size                            (LeType                  type);

const char *       le_type_name                            (LeType                  type);

G_END_DECLS

#endif
