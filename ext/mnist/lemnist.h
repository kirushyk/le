/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <le/le.h>

#ifndef __LEMNIST_H__
#define __LEMNIST_H__

typedef struct MNIST
{
    LeDataSet *train;
    LeDataSet *test;
} MNIST;

MNIST * le_mnist_load (const char *path);

void    le_mnist_free (MNIST      *mnist);

#endif
