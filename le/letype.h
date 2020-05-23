/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LETYPE_H__
#define __LETYPE_H__

#include <stddef.h>
#include <le/lemacros.h>

LE_BEGIN_DECLS

typedef enum LeType
{
    LE_TYPE_VOID,
    LE_TYPE_UINT8,
    LE_TYPE_INT8,
    LE_TYPE_INT16,
    LE_TYPE_INT32,
    LE_TYPE_FLOAT32,
    LE_TYPE_FLOAT64
} LeType;

size_t le_type_size(LeType type);

LE_END_DECLS

#endif
