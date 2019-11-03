/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lemnist.h"
#include <stdlib.h>

MNIST *
le_mnist_load(const char *path)
{
    MNIST *mnist = malloc(sizeof(MNIST));
    mnist->test = NULL;
    mnist->train = NULL;
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
