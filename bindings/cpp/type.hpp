#pragma once
#include <le/le.h>

namespace le
{

enum class Type
{
    VOID = LE_TYPE_VOID,
    U8 = LE_TYPE_U8,
    I8 = LE_TYPE_I8,
    I16 = LE_TYPE_I16,
    I32 = LE_TYPE_I32,
    F32 = LE_TYPE_F32,
    F64 = LE_TYPE_F64
};

}
