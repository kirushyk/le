/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>
#include <ext/mnist/lemnist.h>

int
main()
{
    MNIST *mnist = le_mnist_load(NULL);

    LeTensor *train_images = le_data_set_get_input(mnist->train);
    le_tensor_reshape(train_images, 2, 60000, 28 * 28);
    LeTensor *train_input = le_matrix_new_transpose(train_images);
    LeTensor *train_input_f32 = le_tensor_new_cast(train_input, LE_TYPE_FLOAT32);
    le_tensor_mul(train_input_f32, 2.0f / 255.0f);
    le_tensor_sub(train_input_f32, 1.0f);
    LeTensor *train_labels = le_data_set_get_output(mnist->train);
    le_tensor_reshape(train_labels, 2, 1, 60000);
    LeTensor *train_output = le_matrix_new_one_hot(LE_TYPE_FLOAT32, train_labels, 10);
    LeTensor *test_images = le_data_set_get_input(mnist->test);
    le_tensor_reshape(test_images, 2, 10000, 28 * 28);
    LeTensor *test_input = le_matrix_new_transpose(test_images);
    LeTensor *test_input_f32 = le_tensor_new_cast(test_input, LE_TYPE_FLOAT32);
    le_tensor_mul(test_input_f32, 2.0f / 255.0f);
    le_tensor_sub(test_input_f32, 1.0f);
    LeTensor *test_labels = le_data_set_get_output(mnist->test);
    le_tensor_reshape(test_labels, 2, 1, 10000);
    LeTensor *test_output = le_matrix_new_one_hot(LE_TYPE_FLOAT32, test_labels, 10);
    
    Le1LayerNN *neural_network = le_1_layer_nn_new();
    le_1_layer_nn_init(neural_network, 28 * 28, 10);
    LeBGD *optimizer = le_bgd_new(LE_MODEL(neural_network), train_input_f32, train_output, 0.03f);
    for (unsigned i = 0; i <= 2500; i++)
    {
        le_optimizer_step(LE_OPTIMIZER(optimizer));
        
        if (i % 100 == 0) {
            printf("Iteration %d.\n", i);
            
            LeTensor *train_prediction = le_model_predict(LE_MODEL(neural_network), train_input_f32);
            float train_set_error = le_one_hot_misclassification(train_prediction, train_output);
            printf("Train Set Error: %f\n", train_set_error);
            le_tensor_free(train_prediction);

            LeTensor *test_prediction = le_model_predict(LE_MODEL(neural_network), test_input_f32);
            float test_set_error = le_one_hot_misclassification(test_prediction, test_output);
            printf("Test Set Error: %f\n", test_set_error);
            le_tensor_free(test_prediction);
        }
    }
    
    le_1_layer_nn_free(neural_network);
    le_tensor_free(test_output);
    le_tensor_free(test_input_f32);
    le_tensor_free(test_input);
    le_tensor_free(train_output);
    le_tensor_free(train_input_f32);
    le_tensor_free(train_input);
    le_mnist_free(mnist);

    return EXIT_SUCCESS;
}
