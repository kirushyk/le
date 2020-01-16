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

    assert(mnist);
    assert(mnist->train);
    assert(mnist->test);

    LeTensor *train_images = le_data_set_get_input(mnist->train);

    /// @note: In case MNIST dataset is not installed with
    /// `ninja install`, this test will fail here:
    assert(train_images);
    assert(train_images->element_type == LE_TYPE_UINT8);
    assert(train_images->shape);
    assert(train_images->shape->num_dimensions = 3);
    assert(train_images->shape->sizes[0] = 60000);
    assert(train_images->shape->sizes[1] = 28);
    assert(train_images->shape->sizes[2] = 28);
    assert(train_images->owns_data);
    assert(train_images->data);

    LeTensor *train_labels = le_data_set_get_output(mnist->train);

    assert(train_labels);
    assert(train_labels->element_type == LE_TYPE_UINT8);
    assert(train_labels->shape);
    assert(train_labels->shape->num_dimensions = 1);
    assert(train_labels->shape->sizes[0] = 60000);
    assert(train_labels->owns_data);
    assert(train_labels->data);

    LeTensor *test_images = le_data_set_get_input(mnist->test);

    assert(test_images);
    assert(test_images->element_type == LE_TYPE_UINT8);
    assert(test_images->shape);
    assert(test_images->shape->num_dimensions = 3);
    assert(test_images->shape->sizes[0] = 10000);
    assert(test_images->shape->sizes[1] = 28);
    assert(test_images->shape->sizes[2] = 28);
    assert(test_images->owns_data);
    assert(test_images->data);

    LeTensor *test_labels = le_data_set_get_output(mnist->test);

    assert(test_labels);
    assert(test_labels->element_type == LE_TYPE_UINT8);
    assert(test_labels->shape);
    assert(test_labels->shape->num_dimensions = 1);
    assert(test_labels->shape->sizes[0] = 10000);
    assert(test_labels->owns_data);
    assert(test_labels->data);

    le_mnist_free(mnist);

    return EXIT_SUCCESS;
}
