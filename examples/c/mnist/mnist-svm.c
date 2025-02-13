/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <assert.h>
#include <ext/mnist/lemnist.h>
#include <glib.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>
#include <stdlib.h>

int
main ()
{
  MNIST *mnist = le_mnist_load (NULL);

  LeTensor *train_images = le_data_set_get_input (mnist->train);
  le_tensor_reshape (train_images, 2, 60000, 28 * 28);
  LeTensor *train_input     = le_matrix_new_transpose (train_images);
  LeTensor *train_input_f32 = le_tensor_new_cast (train_input, LE_TYPE_F32);
  le_tensor_mul (train_input_f32, 2.0f / 255.0f);
  le_tensor_sub (train_input_f32, 1.0f);
  LeTensor *train_labels = le_data_set_get_output (mnist->train);
  le_tensor_reshape (train_labels, 2, 1, 60000);
  LeTensor *train_output = le_tensor_new_equal_u8 (LE_TYPE_F32, train_labels, 5);
  LeTensor *test_images  = le_data_set_get_input (mnist->test);
  le_tensor_reshape (test_images, 2, 10000, 28 * 28);
  LeTensor *test_input     = le_matrix_new_transpose (test_images);
  LeTensor *test_input_f32 = le_tensor_new_cast (test_input, LE_TYPE_F32);
  le_tensor_mul (test_input_f32, 2.0f / 255.0f);
  le_tensor_sub (test_input_f32, 1.0f);
  LeTensor *test_labels = le_data_set_get_output (mnist->test);
  le_tensor_reshape (test_labels, 2, 1, 10000);
  LeTensor *test_output = le_tensor_new_equal_u8 (LE_TYPE_F32, test_labels, 5);

  LeSVM               *classifier = le_svm_new ();
  LeSVMTrainingOptions options;
  options.kernel = LE_KERNEL_RBF;
  options.c      = 1;
  le_svm_train (classifier, train_input_f32, train_output, options);
  g_object_unref (classifier);

  le_tensor_unref (test_output);
  le_tensor_unref (test_input_f32);
  le_tensor_unref (test_input);
  le_tensor_unref (train_output);
  le_tensor_unref (train_input_f32);
  le_tensor_unref (train_input);
  le_mnist_free (mnist);

  return EXIT_SUCCESS;
}
