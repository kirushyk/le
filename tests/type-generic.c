/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>

int
main ()
{
  LeTensor *m;

  m = le_tensor_new (LE_TYPE_F32, 2, 1, 1, 1.0);
  le_matrix_set (m, 0, 0, 2.0f);
  le_tensor_unref (m);

  m = le_tensor_new (LE_TYPE_F64, 2, 1, 1, 1.0);
  le_matrix_set (m, 0, 0, 2.0);
  le_tensor_unref (m);

  m = le_tensor_new (LE_TYPE_I8, 2, 1, 1, 1);
  le_matrix_set (m, 0, 0, (gint8)2);
  le_tensor_unref (m);

  m = le_tensor_new (LE_TYPE_U8, 2, 1, 1, 1);
  le_matrix_set (m, 0, 0, (guint8)2);
  le_tensor_unref (m);

  m = le_tensor_new (LE_TYPE_I16, 2, 1, 1, 1);
  le_matrix_set (m, 0, 0, (gint16)2);
  le_tensor_unref (m);

  m = le_tensor_new (LE_TYPE_U16, 2, 1, 1, 1);
  le_matrix_set (m, 0, 0, (guint16)2);
  le_tensor_unref (m);

  m = le_tensor_new (LE_TYPE_I32, 2, 1, 1, 1);
  le_matrix_set (m, 0, 0, (gint32)2);
  le_tensor_unref (m);

  m = le_tensor_new (LE_TYPE_U32, 2, 1, 1, 1);
  le_matrix_set (m, 0, 0, (guint32)2);
  le_tensor_unref (m);

  return EXIT_SUCCESS;
}
