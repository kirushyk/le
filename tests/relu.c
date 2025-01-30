/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <glib.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>

int
main ()
{
  LeTensor *m = le_tensor_new (LE_TYPE_F32, 2, 1, 1, 1.0);
  le_matrix_set (m, 0, 0, 2.0f);
  g_assert_cmpint (le_matrix_at_f32 (m, 0, 0), ==, 2.0f);
  le_tensor_apply_relu (m);
  g_assert_cmpint (le_matrix_at_f32 (m, 0, 0), ==, 2.0f);
  le_matrix_set (m, 0, 0, -2.0f);
  g_assert_cmpint (le_matrix_at_f32 (m, 0, 0), ==, -2.0f);
  le_tensor_apply_relu (m);
  g_assert_cmpint (le_matrix_at_f32 (m, 0, 0), ==, 0);
  le_tensor_unref (m);

  m = le_tensor_new (LE_TYPE_F64, 2, 1, 1, 1.0);
  le_matrix_set (m, 0, 0, 2.0);
  g_assert_cmpint (le_matrix_at_f64 (m, 0, 0), ==, 2.0);
  le_tensor_apply_relu (m);
  g_assert_cmpint (le_matrix_at_f64 (m, 0, 0), ==, 2.0);
  le_matrix_set (m, 0, 0, -2.0);
  g_assert_cmpint (le_matrix_at_f64 (m, 0, 0), ==, -2.0);
  le_tensor_apply_relu (m);
  g_assert_cmpint (le_matrix_at_f64 (m, 0, 0), ==, 0);
  le_tensor_unref (m);

  m = le_tensor_new (LE_TYPE_I8, 2, 1, 1, 1);
  le_matrix_set (m, 0, 0, (gint8)2);
  g_assert_cmpint (le_matrix_at_i8 (m, 0, 0), ==, 2);
  le_tensor_apply_relu (m);
  g_assert_cmpint (le_matrix_at_i8 (m, 0, 0), ==, 2);
  le_matrix_set (m, 0, 0, (gint8)-2);
  g_assert_cmpint (le_matrix_at_i8 (m, 0, 0), ==, -2);
  le_tensor_apply_relu (m);
  g_assert_cmpint (le_matrix_at_i8 (m, 0, 0), ==, 0);
  le_tensor_unref (m);

  m = le_tensor_new (LE_TYPE_I16, 2, 1, 1, 1);
  le_matrix_set (m, 0, 0, (gint16)2);
  g_assert_cmpint (le_matrix_at_i16 (m, 0, 0), ==, 2);
  le_tensor_apply_relu (m);
  g_assert_cmpint (le_matrix_at_i16 (m, 0, 0), ==, 2);
  le_matrix_set (m, 0, 0, (gint16)-2);
  g_assert_cmpint (le_matrix_at_i16 (m, 0, 0), ==, -2);
  le_tensor_apply_relu (m);
  g_assert_cmpint (le_matrix_at_i16 (m, 0, 0), ==, 0);
  le_tensor_unref (m);

  m = le_tensor_new (LE_TYPE_I32, 2, 1, 1, 1);
  le_matrix_set (m, 0, 0, (gint32)2);
  g_assert_cmpint (le_matrix_at_i32 (m, 0, 0), ==, 2);
  le_tensor_apply_relu (m);
  g_assert_cmpint (le_matrix_at_i32 (m, 0, 0), ==, 2);
  le_matrix_set (m, 0, 0, (gint32)-2);
  g_assert_cmpint (le_matrix_at_i32 (m, 0, 0), ==, -2);
  le_tensor_apply_relu (m);
  g_assert_cmpint (le_matrix_at_i32 (m, 0, 0), ==, 0);
  le_tensor_unref (m);

  return EXIT_SUCCESS;
}
