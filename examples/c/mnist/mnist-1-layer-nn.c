/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <assert.h>
#include <ext/mnist/lemnist.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>
#include <stdlib.h>

void
print_case (LeTensor *X, LeTensor *Y, unsigned index, LeModel *model)
{
  LeTensor *x = le_matrix_get_column_copy (X, index);
  LeTensor *y = le_matrix_get_column_copy (Y, index);

  LeTensor *h = le_model_predict (model, x);

  for (unsigned i = 0; i < 28; i++) {
    for (unsigned j = 0; j < 28; j++) {
      char  c[] = " .:-=+*#%%@";
      gsize ix  = le_matrix_at_f32 (x, i * 28 + j, 0) * 10;
      putchar (c[ix]);
    }
    putchar ('\n');
  }

  int    label = -1, pred = -1;
  gfloat label_prob = 0.0f, pred_prob = 0.0f;
  for (unsigned i = 0; i < 10; i++) {
    if (le_matrix_at_f32 (y, i, 0) > label_prob) {
      label_prob = le_matrix_at_f32 (y, i, 0);
      label      = i;
    }
    if (le_matrix_at_f32 (h, i, 0) > pred_prob) {
      pred_prob = le_matrix_at_f32 (h, i, 0);
      pred      = i;
    }
  }

  printf ("%u) Label: %d, Pred: %d (%f)\n", index, label, pred, pred_prob);

  le_tensor_unref (h);
  le_tensor_unref (y);
  le_tensor_unref (x);
}

int
main ()
{
  MNIST *mnist = le_mnist_load (NULL);

  LeTensor *train_images = le_data_set_get_input (mnist->train);
  assert (le_tensor_reshape (train_images, 2, 60000, 28 * 28));
  LeTensor *train_input     = le_matrix_new_transpose (train_images);
  LeTensor *train_input_f32 = le_tensor_new_cast (train_input, LE_TYPE_F32);
  le_tensor_mul (train_input_f32, 2.0f / 255.0f);
  le_tensor_sub (train_input_f32, 1.0f);
  LeTensor *train_labels = le_data_set_get_output (mnist->train);
  assert (le_tensor_reshape (train_labels, 2, 1, 60000));
  LeTensor *train_output = le_matrix_new_one_hot (LE_TYPE_F32, train_labels, 10);
  LeTensor *test_images  = le_data_set_get_input (mnist->test);
  assert (le_tensor_reshape (test_images, 2, 10000, 28 * 28));
  LeTensor *test_input     = le_matrix_new_transpose (test_images);
  LeTensor *test_input_f32 = le_tensor_new_cast (test_input, LE_TYPE_F32);
  le_tensor_mul (test_input_f32, 2.0f / 255.0f);
  le_tensor_sub (test_input_f32, 1.0f);
  LeTensor *test_labels = le_data_set_get_output (mnist->test);
  assert (le_tensor_reshape (test_labels, 2, 1, 10000));
  LeTensor *test_output = le_matrix_new_one_hot (LE_TYPE_F32, test_labels, 10);

  LeSequential *neural_network = le_sequential_new ();
  LeDenseLayer *layer          = le_dense_layer_new ("d1", 28 * 28, 10);
  le_sequential_add (neural_network, LE_LAYER (layer));
  le_sequential_add (neural_network, LE_LAYER (le_activation_layer_new ("a1", LE_ACTIVATION_SOFTMAX)));
  LeLoss loss = LE_LOSS_CROSS_ENTROPY;
  le_sequential_set_loss (neural_network, loss);
  gsize        num_epochs    = 100;
  gsize        batch_size    = 64;
  gfloat       learning_rate = 1e-3f;
  gfloat       momentum      = 0.9f;
  LeOptimizer *optimizer     = LE_OPTIMIZER (
      le_sgd_new (LE_MODEL (neural_network), train_input_f32, train_output, batch_size, learning_rate, momentum));
  for (unsigned i = 0, j = 0; i <= num_epochs * 60000; i += batch_size) {
    le_optimizer_step (optimizer);

    if (i - j >= 60000) {
      j = i;
      printf ("Iteration %d.\n", i);

      LeTensor *train_prediction = le_model_predict (LE_MODEL (neural_network), train_input_f32);

      gfloat train_loss              = le_loss (loss, train_prediction, train_output);
      gfloat train_misclassification = le_one_hot_misclassification (train_prediction, train_output);
      printf ("Train Set Loss: %f, Misclassification: %f\n", train_loss, train_misclassification);
      le_tensor_unref (train_prediction);

      LeTensor *test_prediction = le_model_predict (LE_MODEL (neural_network), test_input_f32);

      gfloat test_loss              = le_loss (loss, test_prediction, test_output);
      gfloat test_misclassification = le_one_hot_misclassification (test_prediction, test_output);
      printf ("Test Set Loss: %f, Misclassification: %f\n", test_loss, test_misclassification);
      le_tensor_unref (test_prediction);
    }
  }

  for (int i = 0; i < 10; i++) {
    print_case (train_input_f32, train_output, i, LE_MODEL (neural_network));
  }

  g_object_unref (LE_SGD (optimizer));
  // g_object_unref(LE_SGD(optimizer));
  g_object_unref (neural_network);
  le_tensor_unref (test_output);
  le_tensor_unref (test_input_f32);
  le_tensor_unref (test_input);
  le_tensor_unref (train_output);
  le_tensor_unref (train_input_f32);
  le_tensor_unref (train_input);
  le_mnist_free (mnist);

  return EXIT_SUCCESS;
}
