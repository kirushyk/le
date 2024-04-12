/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "letype.h"

gsize
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

const char *
le_type_name(LeType type)
{
    switch (type) {
    case LE_TYPE_INT8:
        return "int8";
    case LE_TYPE_UINT8:
        return "uint8";
    case LE_TYPE_INT16:
        return "int16";
    case LE_TYPE_UINT16:
        return "uint16";
    case LE_TYPE_FLOAT16:
        return "float16";
    case LE_TYPE_INT32:
        return "int32";
    case LE_TYPE_UINT32:
        return "uint32";
    case LE_TYPE_FLOAT32:
        return "float32";
    case LE_TYPE_FLOAT64:
        return "float64";
    case LE_TYPE_VOID:
    default:
        return "void";
    }
}
