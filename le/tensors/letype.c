/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "letype.h"

gsize
le_type_size (LeType type)
{
  switch (type) {
  case LE_TYPE_I8:
  case LE_TYPE_U8:
    return 1;
  case LE_TYPE_I16:
  case LE_TYPE_U16:
  case LE_TYPE_F16:
    return 2;
  case LE_TYPE_I32:
  case LE_TYPE_U32:
  case LE_TYPE_F32:
    return 4;
  case LE_TYPE_F64:
    return 8;
  case LE_TYPE_VOID:
  default:
    return 0;
  }
}

const char *
le_type_name (LeType type)
{
  switch (type) {
  case LE_TYPE_I8:
    return "i8";
  case LE_TYPE_U8:
    return "u8";
  case LE_TYPE_I16:
    return "i16";
  case LE_TYPE_U16:
    return "u16";
  case LE_TYPE_F16:
    return "f16";
  case LE_TYPE_I32:
    return "i32";
  case LE_TYPE_U32:
    return "u32";
  case LE_TYPE_F32:
    return "f32";
  case LE_TYPE_F64:
    return "f64";
  case LE_TYPE_VOID:
  default:
    return "void";
  }
}
