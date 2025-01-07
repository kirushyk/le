/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "tests/tensorlist"

#include "test-config.h"
#include <glib.h>
#include <stdlib.h>
#include <le/le.h>
#include <ext/tensorlist/letensorlist.h>

#define TENSORLIST_FILENAME TEST_DIR "/test.tensorlist"

int
main ()
{
  GList *tensorlist = g_list_append (NULL, le_tensor_new (LE_TYPE_UINT8, 2, 2, 2, // clang-format off
        1, 2,
        3, 4
    )); // clang-format on
  tensorlist = g_list_append (tensorlist, le_tensor_new (LE_TYPE_FLOAT32, 2, 3, 3, // clang-format off
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    )); // clang-format on
  tensorlist = g_list_append (tensorlist, le_matrix_new_zeros (LE_TYPE_FLOAT32, 300, 300));
  tensorlist = g_list_append (tensorlist, le_matrix_new_rand_f32 (LE_DISTRIBUTION_UNIFORM, 50, 50));
  tensorlist = g_list_append (tensorlist, le_tensor_new (LE_TYPE_INT16, 2, 5, 5, // clang-format off
        5, 4, 3, 2, 1,
        6, 5, 4, 3, 2,
        7, 6, 5, 4, 3,
        8, 7, 6, 5, 4,
        9, 8, 7, 6, 5
    )); // clang-format on

  LE_INFO ("Saving list of tensors to " TENSORLIST_FILENAME);
  le_tensorlist_save (tensorlist, TENSORLIST_FILENAME);

  LE_INFO ("Loading list of tensors from " TENSORLIST_FILENAME);
  GList *loaded_tensorlist = le_tensorlist_load (TENSORLIST_FILENAME);

  GList *i = NULL, *j = NULL;
  for (i = tensorlist, j = loaded_tensorlist; i && j; i = i->next, j = j->next) {
    LeTensor *original = LE_TENSOR (i->data);
    LeTensor *loaded = LE_TENSOR (j->data);
    LE_INFO ("Comparing original Tensor\n%s", le_tensor_to_cstr (original));
    LE_INFO ("with saved and loaded Tensor\n%s", le_tensor_to_cstr (loaded));
    g_assert_true (le_tensor_equal (original, loaded));
  }

  g_assert_null (i);
  g_assert_null (j);

  g_list_free_full (loaded_tensorlist, (GDestroyNotify)le_tensor_free);
  g_list_free_full (tensorlist, (GDestroyNotify)le_tensor_free);

  return EXIT_SUCCESS;
}
