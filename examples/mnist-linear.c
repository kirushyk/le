/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>
#include <le/letensor-imp.h>
#include <ext/mnist/lemnist.h>

int
main()
{
    MNIST *mnist = le_mnist_load(NULL);

    LeTensor *train_images = le_data_set_get_input(mnist->train);
    le_tensor_reshape(train_images, 2, 60000, 28 * 28);
    LeTensor *train_input = le_matrix_new_transpose(train_images);
    LeTensor *train_labels = le_data_set_get_output(mnist->train);
    le_tensor_reshape(train_labels, 2, 1, 60000);
    LeTensor *test_images = le_data_set_get_input(mnist->test);
    le_tensor_reshape(test_images, 2, 10000, 28 * 28);
    LeTensor *test_input = le_matrix_new_transpose(test_images);
    LeTensor *test_labels = le_data_set_get_output(mnist->test);
    le_tensor_reshape(test_labels, 2, 1, 10000);
    
    le_tensor_free(test_input);
    le_tensor_free(train_input);
    le_mnist_free(mnist);

    return EXIT_SUCCESS;
}
