/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "pg-color.h"

/*
 BGRA32
 color_for_tanh(gfloat scalar)
 {
 BGRA32 color;
 if (scalar > 0)
 {
 color.r = 255;
 color.g = (guint8)((1.f - scalar * 0.5) * 255);
 color.b = (guint8)((1.f - scalar) * 255);
 }
 else
 {
 color.r = (guint8)((scalar + 1.f) * 255);
 color.g = (guint8)((0.5 * scalar + 1.f) * 255);
 color.b = 255;
 }
 color.a = 255;
 return color;
 }
 */

BGRA32
color_for_logistic (gfloat scalar)
{
  BGRA32 color;
  scalar = scalar * 2.f - 1.f;
  if (scalar > 0) {
    color.r = 255;
    color.g = (guint8)((1.f - scalar * 0.5) * 255);
    color.b = (guint8)((1.f - scalar) * 255);
  } else {
    color.r = (guint8)((scalar + 1.f) * 255);
    color.g = (guint8)((0.5 * scalar + 1.f) * 255);
    color.b = 255;
  }
  color.a = 255;
  return color;
}
