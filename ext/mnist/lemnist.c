/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lemnist.h"
#include <mnist-config.h>
#include <stdlib.h>
#include <ext/idx/leidx.h>

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

MNIST *
le_mnist_load (const char *path)
{
  LeTensor *train_input = NULL;
  LeTensor *train_output = NULL;
  LeTensor *test_input = NULL;
  LeTensor *test_output = NULL;

  if (!path) {
    path = MNIST_DATASET_INSTALL_PATH;
  }

  char buffer[1024];

  sprintf (buffer, "%s/train-images-idx3-ubyte.gz", path);
  train_input = le_idx_gz_read (buffer);
  if (!train_input)
    goto error;
  sprintf (buffer, "%s/train-labels-idx1-ubyte.gz", path);
  train_output = le_idx_gz_read (buffer);
  if (!train_output)
    goto error;

  sprintf (buffer, "%s/t10k-images-idx3-ubyte.gz", path);
  test_input = le_idx_gz_read (buffer);
  if (!test_input)
    goto error;
  sprintf (buffer, "%s/t10k-labels-idx1-ubyte.gz", path);
  test_output = le_idx_gz_read (buffer);
  if (!test_output)
    goto error;

  MNIST *mnist = g_new0 (MNIST, 1);
  mnist->train = le_data_set_new_take (train_input, train_output);
  mnist->test = le_data_set_new_take (test_input, test_output);
  return mnist;

error:
  if (train_input) {
    le_tensor_unref (train_input);
  }
  if (train_output) {
    le_tensor_unref (train_output);
  }
  if (test_input) {
    le_tensor_unref (test_input);
  }
  if (test_output) {
    le_tensor_unref (test_output);
  }
  return NULL;
}

void
le_mnist_free (MNIST *mnist)
{
  if (mnist) {
    le_data_set_free (mnist->train);
    le_data_set_free (mnist->test);
    g_free (mnist);
  }
}
