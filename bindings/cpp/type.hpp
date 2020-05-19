#pragma once
#include <le/le.h>

namespace le
{

enum class Type
{
    VOID = LE_TYPE_VOID,
    UINT8 = LE_TYPE_UINT8,
    INT8 = LE_TYPE_INT8,
    INT16 = LE_TYPE_INT16,
    INT32 = LE_TYPE_INT32,
    FLOAT32 = LE_TYPE_FLOAT32,
    FLOAT64 = LE_TYPE_FLOAT64
};

}
