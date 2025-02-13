/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <le/le.h>
#include <stdio.h>

int
main (int argc, const char *argv[])
{
  LeTensor *a = le_matrix_new_rand_f32 (LE_DISTRIBUTION_UNIFORM, 2, 4);
  printf ("a =\n");
  le_tensor_print (a, stdout);

  LeTensor *b = le_matrix_new_polynomia (a);
  printf ("poly(a) =\n");
  le_tensor_print (b, stdout);

  LeTensor *c = le_matrix_new_polynomia (b);
  printf ("poly(poly(a)) =\n");
  le_tensor_print (c, stdout);

  le_tensor_unref (c);
  le_tensor_unref (b);
  le_tensor_unref (a);

  return 0;
}
