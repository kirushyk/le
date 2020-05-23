/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "letype.h"

size_t
le_type_size(LeType type)
{
    switch (type) {
    case LE_TYPE_INT8:
    case LE_TYPE_UINT8:
        return 1;
    case LE_TYPE_INT16:
    case LE_TYPE_UINT16:
    case LE_TYPE_FLOAT16:
        return 2;
    case LE_TYPE_INT32:
    case LE_TYPE_UINT32:
    case LE_TYPE_FLOAT32:
        return 4;
    case LE_TYPE_FLOAT64:
        return 8;
    case LE_TYPE_VOID:
    default:
        return 0;
    }
}
