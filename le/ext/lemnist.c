/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lemnist.h"
#include <stdlib.h>
#include "leidx.h"

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

MNIST *
le_mnist_load(const char *path)
{
    MNIST *mnist = malloc(sizeof(MNIST));
    
    LeMatrix *input = le_idx_read("t10k-images-idx3-ubyte");
    LeMatrix *output = le_idx_read("t10k-labels-idx1-ubyte");
    mnist->train = le_data_set_new_take(input, output);
    
    mnist->test = NULL;
    
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
