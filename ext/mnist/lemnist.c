/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lemnist.h"
#include <stdlib.h>
#include "leidx.h"
#include <mnist-config.h>

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

MNIST *
le_mnist_load(const char *path)
{
    if (!path)
    {
        path = MNIST_DATASET_INSTALL_PATH;
    }

    MNIST *mnist = malloc(sizeof(MNIST));
    char buffer[1024];
    
    sprintf(buffer, "%s/train-images-idx3-ubyte.gz", path);
    LeTensor *input = le_idx_gz_read(buffer);
    sprintf(buffer, "%s/train-labels-idx1-ubyte.gz", path);
    LeTensor *output = le_idx_gz_read(buffer);
    mnist->train = le_data_set_new_take(input, output);

    sprintf(buffer, "%s/t10k-images-idx3-ubyte.gz", path);
    input = le_idx_gz_read(buffer);
    sprintf(buffer, "%s/t10k-labels-idx1-ubyte.gz", path);
    output = le_idx_gz_read(buffer);
    mnist->test = le_data_set_new_take(input, output);
    
    return mnist;
}

void
le_mnist_free(MNIST *mnist)
{
    if (mnist)
    {
        le_data_set_free(mnist->train);
        le_data_set_free(mnist->test);
        free(mnist);
    }
}
