/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <assert.h>
#include <stdlib.h>
#include <le/le.h>

int
le_test_ensure_matrix_size (LeTensor *a, unsigned height, unsigned width)
{
  if (a == NULL) {
    fprintf (stderr, "NULL pointer given");
    return 0;
  }

  if (le_matrix_get_width (a) != width) {
    fprintf (stderr, "Wrong matrix width.\n");
    return 0;
  }

  if (le_matrix_get_height (a) != height) {
    fprintf (stderr, "Wrong matrix height.\n");
    return 0;
  }

  return 1;
}

#define MAX_DIMENSION 4

int
main ()
{
  unsigned width;
  unsigned height;
  unsigned second_width;

  LeTensor *a, *at, *b, *bt, *c, *d, *ab, *ab_check, *ba, *ba_check;

  a = le_tensor_new (LE_TYPE_FLOAT32, 2, 3, 2, // clang-format off
      1.0, 2.0,
      3.0, 4.0,
      5.0, 6.0
  ); // clang-format on

  b = le_tensor_new (LE_TYPE_FLOAT32, 2, 2, 3, // clang-format off
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0
  ); // clang-format on

  ab_check = le_tensor_new (LE_TYPE_FLOAT32, 2, 3, 3, // clang-format off
      9.0,  12.0, 15.0,
      19.0, 26.0, 33.0,
      29.0, 40.0, 51.0
  ); // clang-format on

  ab = le_matrix_new_product (a, b);
  printf ("ab = ");
  le_tensor_print (ab, stdout);
  printf ("ab_check = ");
  le_tensor_print (ab_check, stdout);
  assert (le_tensor_equal (ab, ab_check));
  le_tensor_unref (ab);

  bt = le_matrix_new_transpose (b);
  ab = le_matrix_new_product_full (a, false, bt, true);
  printf ("ab (of t) = ");
  le_tensor_print (ab, stdout);
  printf ("ab_check = ");
  le_tensor_print (ab_check, stdout);
  printf ("sad afbt %f\n", le_tensor_sad_f32 (ab, ab_check));
  assert (le_tensor_sad_f32 (ab, ab_check) < 1e-3f);
  le_tensor_unref (ab);
  le_tensor_unref (bt);

  at = le_matrix_new_transpose (a);
  ab = le_matrix_new_product_full (at, true, b, false);
  printf ("ab (of t) = ");
  le_tensor_print (ab, stdout);
  printf ("ab_check = ");
  le_tensor_print (ab_check, stdout);
  printf ("sad atbf %f\n", le_tensor_sad_f32 (ab, ab_check));
  assert (le_tensor_sad_f32 (ab, ab_check) < 1e-3f);
  le_tensor_unref (ab);
  le_tensor_unref (at);

  le_tensor_unref (ab_check);

  ba = le_matrix_new_product (b, a);
  ba_check = le_tensor_new (LE_TYPE_FLOAT32, 2, 2, 2, // clang-format off
      22.0, 28.0,
      49.0, 64.0
  ); // clang-format on
  assert (le_tensor_equal (ba, ba_check));
  le_tensor_unref (ba_check);
  le_tensor_unref (ba);

  le_tensor_unref (b);
  le_tensor_unref (a);

  for (height = 1; height < MAX_DIMENSION; height++) {
    a = le_matrix_new_identity (LE_TYPE_FLOAT32, height);
    assert (le_test_ensure_matrix_size (a, height, height));
    le_tensor_unref (a);

    for (width = 1; width < MAX_DIMENSION; width++) {
      a = le_matrix_new_zeros (LE_TYPE_FLOAT32, height, width);
      assert (le_test_ensure_matrix_size (a, height, width));
      le_tensor_unref (a);

      a = le_matrix_new_rand_f32 (LE_DISTRIBUTION_UNIFORM, height, width);
      assert (le_test_ensure_matrix_size (a, height, width));

      for (second_width = 1; second_width < MAX_DIMENSION; second_width++) {
        b = le_matrix_new_rand_f32 (LE_DISTRIBUTION_UNIFORM, width, second_width);
        c = le_matrix_new_product (a, b);
        assert (le_test_ensure_matrix_size (c, height, second_width));
        le_tensor_unref (c);
        le_tensor_unref (b);
      }

      le_tensor_unref (a);
    }
  }

  a = le_matrix_new_rand_f32 (LE_DISTRIBUTION_UNIFORM, 10, 5);
  b = le_matrix_new_rand_f32 (LE_DISTRIBUTION_UNIFORM, 10, 5);
  at = le_matrix_new_transpose (a);
  c = le_matrix_new_product (at, b);
  d = le_matrix_new_product_full (a, true, b, false);
  printf ("c = ");
  le_tensor_print (c, stdout);
  printf ("d = ");
  le_tensor_print (d, stdout);
  printf ("sad atbf %f\n", le_tensor_sad_f32 (c, d));
  assert (le_tensor_sad_f32 (c, d) < 1e-3f);
  le_tensor_unref (d);
  le_tensor_unref (c);
  le_tensor_unref (at);
  bt = le_matrix_new_transpose (b);
  c = le_matrix_new_product (a, bt);
  d = le_matrix_new_product_full (a, false, b, true);
  printf ("sad afbt %f\n", le_tensor_sad_f32 (c, d));
  assert (le_tensor_sad_f32 (c, d) < 1e-3f);
  le_tensor_unref (bt);
  le_tensor_unref (b);
  le_tensor_unref (a);

  return EXIT_SUCCESS;
}
