/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lepolynomia.h"
#include <glib.h>
#include <le/tensors/lematrix.h>
#include <le/tensors/letensor-imp.h>

LeTensor *
le_matrix_new_polynomia (const LeTensor *a)
{
  /// @todo: Check tensor device
  g_assert_cmpint (a->element_type, ==, LE_TYPE_F32);

  /// @todo: Use proper types 
  int initial_features_count    = le_matrix_get_height (a);
  int additional_features_count = initial_features_count * (initial_features_count + 1) / 2;
  int examples_count            = le_matrix_get_width (a);

  LeTensor *polynomia =
      le_matrix_new_uninitialized (a->element_type, initial_features_count + additional_features_count, examples_count);
  for (gint example = 0; example < examples_count; example++) {
    for (gint feature = 0; feature < initial_features_count; feature++) {
      le_matrix_set (polynomia, feature, example, le_matrix_at_f32 (a, feature, example));
    }

    int additional_feature_index = initial_features_count;
    for (gint feature = 0; feature < initial_features_count; feature++) {
      for (gint another_feature = feature; another_feature < initial_features_count; another_feature++) {
        gfloat additional_feature =
            le_matrix_at_f32 (a, feature, example) * le_matrix_at_f32 (a, another_feature, example);
        le_matrix_set (polynomia, additional_feature_index, example, additional_feature);
        additional_feature_index++;
      }
    }
  }

  return polynomia;
}
