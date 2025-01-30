
/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>

int
main()
{
  LeTensor *a = le_tensor_new (LE_TYPE_F32, 2, 3, 1,
    2.0,
    5.0,
    8.0
  );
  LeTensor *b = le_tensor_new (LE_TYPE_F32, 2, 1, 3,
    1.0, 2.0, 3.0
  );
  LeTensor *greater_a = le_tensor_new (LE_TYPE_F32, 2, 3, 5,
    0.0, 1.0, 2.0, 3.0, 0.0,
    0.0, 4.0, 5.0, 6.0, 0.0,
    0.0, 7.0, 8.0, 9.0, 0.0
  );
  LeTensor *a_as_view = le_matrix_get_column (greater_a, 2);
  assert (le_tensor_equal (a, a_as_view));
  LeTensor *c1 = le_matrix_new_product (b, a);
  LeTensor *c2 = le_matrix_new_product (b, a_as_view);
  assert (le_tensor_equal (c1, c2));
  le_tensor_unref (c2);
  le_tensor_unref (c1);
  LeTensor *c3 = le_matrix_new_product_full (a, true, b, true);
  LeTensor *c4 = le_matrix_new_product_full (a_as_view, true, b, true);
  assert (le_tensor_equal (c3, c4));
  le_tensor_unref (c4);
  le_tensor_unref (c3);
  le_tensor_unref (a_as_view);
  le_tensor_unref (greater_a);
  le_tensor_unref (b);
  le_tensor_unref (a);
  return EXIT_SUCCESS;
}
