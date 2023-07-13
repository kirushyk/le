
/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>

int
main()
{
  LeTensor *a = le_tensor_new (LE_TYPE_FLOAT32, 2, 4, 3,
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    1.0, 2.0, 3.0,
  );
  LeTensor *a_t = le_tensor_new (LE_TYPE_FLOAT32, 2, 3, 4,
    1.0, 4.0, 7.0, 1.0,
    2.0, 5.0, 8.0, 2.0,
    3.0, 6.0, 9.0, 3.0
  );
  LeTensor *greater_a = le_tensor_new (LE_TYPE_FLOAT32, 2, 4, 5,
    0.0, 1.0, 2.0, 3.0, 0.0,
    0.0, 4.0, 5.0, 6.0, 0.0,
    0.0, 7.0, 8.0, 9.0, 0.0,
    0.0, 1.0, 2.0, 3.0, 0.0
  );
  LeTensor *a_as_view = le_matrix_get_columns_copy (greater_a, 1, 3);
  assert (le_tensor_equal (a, a_as_view));
  LeTensor *a_as_view_t = le_matrix_new_transpose (a_as_view);
  LeTensor *a_t_computed = le_matrix_new_transpose (a);
  assert (le_tensor_equal (a_as_view_t, a_t));
  assert (le_tensor_equal (a_as_view_t, a_t_computed));
  le_tensor_free (a_as_view_t);
  le_tensor_free (a_t_computed);
  le_tensor_free (a_as_view);
  le_tensor_free (greater_a);
  le_tensor_free (a_t);
  le_tensor_free (a);
  return EXIT_SUCCESS;
}
