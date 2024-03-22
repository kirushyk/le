/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>
#include <ext/mnist/lemnist.h>
#include <ext/tensorlist/letensorlist.h>

#define DEFAULT_LOG_CATEGORY "mnist-snn"

static volatile sig_atomic_t should_quit = 0;

void
on_int_signal(int dummy)
{
    if (should_quit)
    {
        exit(0);
    }
    fprintf(stderr, "\nScheduling quit after current optimization step done\n");
    should_quit = 1;
}

int
main(int argc, char *argv[])
{
    signal(SIGINT, on_int_signal);

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
                      LE_LAYER(le_dense_layer_new("D1", 28 * 28, 800)));
    le_sequential_add(neural_network,
                      LE_LAYER(le_activation_layer_new("A1", LE_ACTIVATION_TANH)));
    le_sequential_add(neural_network,
                      LE_LAYER(le_dense_layer_new("D2", 800, 10)));
    le_sequential_add(neural_network,
                      LE_LAYER(le_activation_layer_new("Softmax", LE_ACTIVATION_SOFTMAX)));

    le_sequential_set_loss(neural_network, LE_LOSS_CROSS_ENTROPY);

    le_sequential_to_dot(neural_network, "2nn.dot");

    LeOptimizer *optimizer = NULL;
    unsigned print_module = 1;
    unsigned num_epochs = 1;
    const char *filename = NULL;

    if (argc >= 2)
    {
        if (strcmp(argv[1], "bgd") == 0)
        {
            optimizer = LE_OPTIMIZER(le_bgd_new(LE_MODEL(neural_network), train_input_f32, train_output, 0.03f));
            print_module = 10;
            num_epochs = 250;
        }
        else if (strcmp(argv[1], "sgd") == 0)
        {
            optimizer = LE_OPTIMIZER(le_sgd_new(LE_MODEL(neural_network), train_input_f32, train_output, 64, 0.03f, 0.9f));
            print_module = 100;
            num_epochs = 2;
        }
        if (argc >= 3)
        {
            filename = argv[2];
        }
    }

    if (filename)
    {
        GList *parameters_iterator = NULL;
        GList *parameters = le_model_get_parameters(LE_MODEL(neural_network));
        GList *loaded_parameters_iterator = NULL;
        GList *loaded_parameters = le_tensorlist_load(filename);
        
        if (loaded_parameters)
        {
            for (parameters_iterator = parameters,
                    loaded_parameters_iterator = loaded_parameters;
                parameters_iterator && loaded_parameters_iterator;
                parameters_iterator = parameters_iterator->next,
                    loaded_parameters_iterator = loaded_parameters_iterator->next)
            {
                LeTensor *parameter = LE_TENSOR(parameters_iterator->data);
                LeTensor *loaded_parameter = LE_TENSOR(loaded_parameters_iterator->data);
                if (le_shape_equal(parameter->shape, loaded_parameter->shape) &&
                    parameter->element_type == loaded_parameter->element_type)
                {
                    le_tensor_assign(parameter, loaded_parameter);
                }
                else
                {
                    LE_ERROR("Incorrect shape or element type of loaded parameter");
                    break;
                }
            }

            if (parameters_iterator || loaded_parameters_iterator)
            {
                LE_ERROR("Incorrect number of loaded parameters");
            }

            printf("Successfully loaded parameters fom %s", filename);
        }
    }

    if (optimizer)
    {
        for (unsigned i = 0; (i <= num_epochs) && !should_quit; i++)
        {
            le_optimizer_epoch(optimizer);
            
            if (i % print_module == 0) 
            {
                if (i > 0)
                    printf("\n");

                printf("Iteration %d\n", i);
                
                LeTensor *train_prediction = le_model_predict(LE_MODEL(neural_network), train_input_f32);
                float train_set_error = le_cross_entropy_loss(train_prediction, train_output);
                float train_set_accuracy = 1.0f - le_one_hot_misclassification(train_prediction, train_output);
                printf("Train Set Error: %f, Accuracy: %.1f%%\n", train_set_error, train_set_accuracy * 100.0f);
                le_tensor_free(train_prediction);

                LeTensor *test_prediction = le_model_predict(LE_MODEL(neural_network), test_input_f32);
                float test_set_error = le_cross_entropy_loss(test_prediction, test_output);
                float test_set_accuracy = 1.0f - le_one_hot_misclassification(test_prediction, test_output);
                printf("Test Set Error: %f, Accuracy: %.1f%%\n", test_set_error, test_set_accuracy * 100.0f);
                le_tensor_free(test_prediction);
            }
            else
            {
                putc('.', stdout);
                fflush(stdout);
            }
        }
    }
    else
    {
        LE_ERROR("No Optimizer specified");
    }

    if (filename)
    {
        le_tensorlist_save(le_model_get_parameters(LE_MODEL(neural_network)), filename);
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
