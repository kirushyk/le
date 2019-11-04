/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LETYPE_H__
#define __LETYPE_H__

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

#endif
