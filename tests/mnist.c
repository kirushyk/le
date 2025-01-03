/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <ext/mnist/mnist-config.h>
#include <stdlib.h>
#include <glib.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>
#include <ext/mnist/lemnist.h>

int
main ()
{
  MNIST *mnist = le_mnist_load (MNIST_DATASET_SOURCE_PATH);

  g_assert_nonnull (mnist);
  g_assert_nonnull (mnist->train);
  g_assert_nonnull (mnist->test);

  LeTensor *train_images = le_data_set_get_input (mnist->train);

  /// @note: In case MNIST dataset is not installed with
  /// `ninja install`, this test will fail here:
  g_assert_nonnull (train_images);
  g_assert_cmpint (train_images->element_type, ==, LE_TYPE_UINT8);
  g_assert_nonnull (train_images->shape);
  g_assert_cmpint (train_images->shape->num_dimensions, ==, 3);
  g_assert_cmpint (train_images->shape->sizes[0], ==, 60000);
  g_assert_cmpint (train_images->shape->sizes[1], ==, 28);
  g_assert_cmpint (train_images->shape->sizes[2], ==, 28);
  g_assert_true (train_images->owns_data);
  g_assert_nonnull (train_images->data);

  LeTensor *train_labels = le_data_set_get_output (mnist->train);

  g_assert_nonnull (train_labels);
  g_assert_cmpint (train_labels->element_type, ==, LE_TYPE_UINT8);
  g_assert_nonnull (train_labels->shape);
  g_assert_cmpint (train_labels->shape->num_dimensions, ==, 1);
  g_assert_cmpint (train_labels->shape->sizes[0], ==, 60000);
  g_assert_true (train_labels->owns_data);
  g_assert_nonnull (train_labels->data);

  LeTensor *test_images = le_data_set_get_input (mnist->test);

  g_assert_nonnull (test_images);
  g_assert_cmpint (test_images->element_type, ==, LE_TYPE_UINT8);
  g_assert_nonnull (test_images->shape);
  g_assert_cmpint (test_images->shape->num_dimensions, ==, 3);
  g_assert_cmpint (test_images->shape->sizes[0], ==, 10000);
  g_assert_cmpint (test_images->shape->sizes[1], ==, 28);
  g_assert_cmpint (test_images->shape->sizes[2], ==, 28);
  g_assert_true (test_images->owns_data);
  g_assert_nonnull (test_images->data);

  LeTensor *test_labels = le_data_set_get_output (mnist->test);

  g_assert_nonnull (test_labels);
  g_assert_cmpint (test_labels->element_type, ==, LE_TYPE_UINT8);
  g_assert_nonnull (test_labels->shape);
  g_assert_cmpint (test_labels->shape->num_dimensions, ==, 1);
  g_assert_cmpint (test_labels->shape->sizes[0], ==, 10000);
  g_assert_true (test_labels->owns_data);
  g_assert_nonnull (test_labels->data);

  le_mnist_free (mnist);

  return EXIT_SUCCESS;
}
