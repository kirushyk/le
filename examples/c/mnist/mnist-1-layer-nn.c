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
    assert(le_tensor_reshape(train_images, 2, 60000, 28 * 28));
    LeTensor *train_input = le_matrix_new_transpose(train_images);
    LeTensor *train_input_f32 = le_tensor_new_cast(train_input, LE_TYPE_FLOAT32);
    le_tensor_mul(train_input_f32, 1.0f / 255.0f);
    // le_tensor_sub(train_input_f32, 1.0f);
    LeTensor *train_labels = le_data_set_get_output(mnist->train);
    le_tensor_reshape(train_labels, 2, 1, 60000);
    LeTensor *train_output = le_matrix_new_one_hot(LE_TYPE_FLOAT32, train_labels, 10);
    LeTensor *test_images = le_data_set_get_input(mnist->test);
    le_tensor_reshape(test_images, 2, 10000, 28 * 28);
    LeTensor *test_input = le_matrix_new_transpose(test_images);
    LeTensor *test_input_f32 = le_tensor_new_cast(test_input, LE_TYPE_FLOAT32);
    le_tensor_mul(test_input_f32, 1.0f / 255.0f);
    // le_tensor_sub(test_input_f32, 1.0f);
    LeTensor *test_labels = le_data_set_get_output(mnist->test);
    le_tensor_reshape(test_labels, 2, 1, 10000);
    LeTensor *test_output = le_matrix_new_one_hot(LE_TYPE_FLOAT32, test_labels, 10);
    
    Le1LayerNN *neural_network = le_1_layer_nn_new();
    le_1_layer_nn_init(neural_network, 28 * 28, 10);
    LeOptimizer *optimizer = LE_OPTIMIZER(le_sgd_new(LE_MODEL(neural_network), train_input_f32, train_output, 60000, 0.03f, 0.0f));
    // LeOptimizer *optimizer = LE_OPTIMIZER(le_bgd_new(LE_MODEL(neural_network), train_input_f32, train_output, 0.003f));
    // Le1LayerNNTrainingOptions options;
    // options.learning_rate = 0.03;
    // options.max_iterations = 1000;
    // le_1_layer_nn_train(neural_network, train_input_f32, train_output, options);
    for (unsigned i = 0; i <= 100 * 60000; i++)
    {
        le_optimizer_step(optimizer);
        
        if (i % 100 == 0) {
            printf("Iteration %d.\n", i);
            
            
            LeTensor *train_prediction = le_model_predict(LE_MODEL(neural_network), train_input_f32);
            // printf("Train Prediction %s:\n%s", le_shape_to_cstr(train_prediction->shape), le_tensor_to_cstr(train_prediction));
            // LeTensorStats test_pred_stats = le_tensor_get_stats(train_prediction);
            // printf("Train Prediction stats:\tmin: %f\tmax: %f\tmean: %f\tdeviation: %f\t nans: %u\t zeros: %u\n",
            //     test_pred_stats.min, test_pred_stats.max, test_pred_stats.mean, test_pred_stats.deviation,
            //     test_pred_stats.nans, test_pred_stats.zeros);


            float train_loss = le_cross_entropy_loss(train_prediction, train_output);
            float train_misclassification = le_one_hot_misclassification(train_prediction, train_output);
            printf("Train Set Loss: %f, Misclassification: %f\n", train_loss, train_misclassification);
            le_tensor_free(train_prediction);

            LeTensor *test_prediction = le_model_predict(LE_MODEL(neural_network), test_input_f32);
            // printf("Test Prediction %s:\n%s", le_shape_to_cstr(test_prediction->shape), le_tensor_to_cstr(test_prediction));
            // LeTensorStats test_pred_stats = le_tensor_get_stats(test_prediction);
            // printf("Test Prediction stats:\tmin: %f\tmax: %f\tmean: %f\tdeviation: %f\t nans: %u\t zeros: %u\n",
            //     test_pred_stats.min, test_pred_stats.max, test_pred_stats.mean, test_pred_stats.deviation,
            //     test_pred_stats.nans, test_pred_stats.zeros);

            float test_loss = le_cross_entropy_loss(test_prediction, test_output);
            float test_misclassification = le_one_hot_misclassification(test_prediction, test_output);
            printf("Test Set Loss: %f, Misclassification: %f\n", test_loss, test_misclassification);
            le_tensor_free(test_prediction);
        }
    }

    le_sgd_free(LE_SGD(optimizer));
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
