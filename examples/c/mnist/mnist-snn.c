/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>
#include <ext/mnist/lemnist.h>

#define DEFAULT_LOG_CATEGORY "mnist-snn"

int
main(int argc, char *argv[])
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
    
    LeSequential *neural_network = le_sequential_new();
    le_sequential_add(neural_network,
                      LE_LAYER(le_dense_layer_new("FC_1", 28 * 28, 300)));
    le_sequential_add(neural_network,
                      LE_LAYER(le_activation_layer_new("ReLU", LE_ACTIVATION_RELU)));
    le_sequential_add(neural_network,
                      LE_LAYER(le_dense_layer_new("FC_2", 300, 10)));
    le_sequential_add(neural_network,
                      LE_LAYER(le_activation_layer_new("Softmax", LE_ACTIVATION_SOFTMAX)));

    le_sequential_set_loss(neural_network, LE_LOSS_CROSS_ENTROPY);

    le_sequential_to_dot(neural_network, "2nn.dot");

    LeOptimizer *optimizer = NULL;
    unsigned print_module = 1;
    unsigned num_epochs = 2500;

    if (argc == 2)
    {
        if (strcmp(argv[1], "bgd") == 0)
        {
            optimizer = LE_OPTIMIZER(le_bgd_new(LE_MODEL(neural_network), train_input_f32, train_output, 0.1f));
            print_module = 1;
            num_epochs = 250;
        }
        else if (strcmp(argv[1], "sgd") == 0)
        {
            optimizer = LE_OPTIMIZER(le_sgd_new(LE_MODEL(neural_network), train_input_f32, train_output, 0.1f, 0.9f));
            print_module = 100;
            num_epochs = 2500;
        }
    }

    if (optimizer)
    {
        for (unsigned i = 0; i <= num_epochs; i++)
        {
            le_optimizer_step(LE_OPTIMIZER(optimizer));
            
            if (i % print_module == 0) {
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
    }
    else
    {
        LE_ERROR("No Optimizer specified");
    }
    

    le_sequential_free(neural_network);
    le_tensor_free(test_output);
    le_tensor_free(test_input_f32);
    le_tensor_free(test_input);
    le_tensor_free(train_output);
    le_tensor_free(train_input_f32);
    le_tensor_free(train_input);
    le_mnist_free(mnist);

    return EXIT_SUCCESS;
}
