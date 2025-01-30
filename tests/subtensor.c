/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "tests/subtensor"

#include "test-config.h"
#include <stdlib.h>
#include <glib.h>
#include <le/le.h>

int
main ()
{
  LeTensor *tensor = le_tensor_new (LE_TYPE_U32, 2, 3, 3, // clang-format off
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  ); // clang-format on

  LeTensor *subtensor = le_tensor_pick (tensor, 1);
  g_assert_cmpint (le_tensor_at_u32 (subtensor, 0), ==, 4);
  g_assert_cmpint (le_tensor_at_u32 (subtensor, 1), ==, 5);
  g_assert_cmpint (le_tensor_at_u32 (subtensor, 2), ==, 6);
  LeTensor *subtensor_copy = le_tensor_new_copy (subtensor);
  g_assert_cmpint (le_tensor_at_u32 (subtensor_copy, 0), ==, 4);
  g_assert_cmpint (le_tensor_at_u32 (subtensor_copy, 1), ==, 5);
  g_assert_cmpint (le_tensor_at_u32 (subtensor_copy, 2), ==, 6);
  g_assert_true (le_tensor_equal (subtensor, subtensor_copy));
  LeTensor *subtensor_direct_copy = le_tensor_pick_copy (tensor, 1);
  g_assert_cmpint (le_tensor_at_u32 (subtensor_direct_copy, 0), ==, 4);
  g_assert_cmpint (le_tensor_at_u32 (subtensor_direct_copy, 1), ==, 5);
  g_assert_cmpint (le_tensor_at_u32 (subtensor_direct_copy, 2), ==, 6);
  g_assert_true (le_tensor_equal (subtensor, subtensor_direct_copy));
  g_assert_true (le_tensor_equal (subtensor_direct_copy, subtensor_copy));
  le_tensor_unref (subtensor_direct_copy);
  le_tensor_unref (subtensor_copy);
  le_tensor_unref (subtensor);

  LeTensor *middle_column = le_matrix_get_column (tensor, 1);
  g_assert_cmpint (le_matrix_at_u32 (middle_column, 0, 0), ==, 2);
  g_assert_cmpint (le_matrix_at_u32 (middle_column, 1, 0), ==, 5);
  g_assert_cmpint (le_matrix_at_u32 (middle_column, 2, 0), ==, 8);
  g_assert_cmpint (le_tensor_at_u32 (middle_column, 0), ==, 2);
  g_assert_cmpint (le_tensor_at_u32 (middle_column, 1), ==, 5);
  g_assert_cmpint (le_tensor_at_u32 (middle_column, 2), ==, 8);
  LeTensor *middle_column_copy = le_tensor_new_copy (middle_column);
  g_assert_cmpint (le_matrix_at_u32 (middle_column_copy, 0, 0), ==, 2);
  g_assert_cmpint (le_matrix_at_u32 (middle_column_copy, 1, 0), ==, 5);
  g_assert_cmpint (le_matrix_at_u32 (middle_column_copy, 2, 0), ==, 8);
  g_assert_cmpint (le_tensor_at_u32 (middle_column_copy, 0), ==, 2);
  g_assert_cmpint (le_tensor_at_u32 (middle_column_copy, 1), ==, 5);
  g_assert_cmpint (le_tensor_at_u32 (middle_column_copy, 2), ==, 8);
  g_assert_true (le_tensor_equal (middle_column, middle_column_copy));
  LeTensor *middle_column_direct_copy = le_matrix_get_column (tensor, 1);
  g_assert_cmpint (le_matrix_at_u32 (middle_column_direct_copy, 0, 0), ==, 2);
  g_assert_cmpint (le_matrix_at_u32 (middle_column_direct_copy, 1, 0), ==, 5);
  g_assert_cmpint (le_matrix_at_u32 (middle_column_direct_copy, 2, 0), ==, 8);
  g_assert_cmpint (le_tensor_at_u32 (middle_column_direct_copy, 0), ==, 2);
  g_assert_cmpint (le_tensor_at_u32 (middle_column_direct_copy, 1), ==, 5);
  g_assert_cmpint (le_tensor_at_u32 (middle_column_direct_copy, 2), ==, 8);
  g_assert_true (le_tensor_equal (middle_column, middle_column_direct_copy));
  g_assert_true (le_tensor_equal (middle_column_copy, middle_column_direct_copy));
  LeTensor *transposed_column = le_matrix_new_transpose (middle_column);
  g_assert_cmpint (le_matrix_at_u32 (transposed_column, 0, 0), ==, 2);
  g_assert_cmpint (le_matrix_at_u32 (transposed_column, 0, 1), ==, 5);
  g_assert_cmpint (le_matrix_at_u32 (transposed_column, 0, 2), ==, 8);
  g_assert_cmpint (le_tensor_at_u32 (transposed_column, 0), ==, 2);
  g_assert_cmpint (le_tensor_at_u32 (transposed_column, 1), ==, 5);
  g_assert_cmpint (le_tensor_at_u32 (transposed_column, 2), ==, 8);
  LeTensor *twice_transposed_column = le_matrix_new_transpose (transposed_column);
  g_assert_cmpint (le_matrix_at_u32 (twice_transposed_column, 0, 0), ==, 2);
  g_assert_cmpint (le_matrix_at_u32 (twice_transposed_column, 1, 0), ==, 5);
  g_assert_cmpint (le_matrix_at_u32 (twice_transposed_column, 2, 0), ==, 8);
  g_assert_cmpint (le_tensor_at_u32 (twice_transposed_column, 0), ==, 2);
  g_assert_cmpint (le_tensor_at_u32 (twice_transposed_column, 1), ==, 5);
  g_assert_cmpint (le_tensor_at_u32 (twice_transposed_column, 2), ==, 8);
  g_assert_true (le_tensor_equal (twice_transposed_column, middle_column));
  g_assert_true (le_tensor_equal (twice_transposed_column, middle_column_copy));
  g_assert_true (le_tensor_equal (twice_transposed_column, middle_column_direct_copy));
  le_tensor_unref (twice_transposed_column);
  le_tensor_unref (transposed_column);
  le_tensor_unref (middle_column_direct_copy);
  le_tensor_unref (middle_column_copy);
  le_tensor_unref (middle_column);

  le_tensor_unref (tensor);
  return 0;
}
