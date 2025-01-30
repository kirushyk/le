/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "snn"

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <le/le.h>

int
main(int argc, const char *argv[])
{
#ifdef __APPLE__
    sranddev();
#else
    srand(time(NULL));
#endif

    LeTensor *x = le_tensor_new(LE_TYPE_F32, 2, 2, 4,
        1.0, 2.0, 1.0, 2.0,
        2.0, 2.0, 1.0, 1.0
    );
    
    /// @note: Non-linearly-separable / XOR-like labels
    LeTensor *y = le_tensor_new(LE_TYPE_F32, 2, 1, 4, 
        0.0, 1.0, 1.0, 0.0
    );
    
    printf("Train set:\n");
    printf("x =\n");
    le_tensor_print(x, stdout);
    printf("y =\n");
    le_tensor_print(y, stdout);

    LE_INFO("Creating Neural Network Structure");

    LeSequential *neural_network = le_sequential_new();
    le_sequential_add(neural_network,
                      LE_LAYER(le_dense_layer_new("D1", 2, 2)));
    le_sequential_add(neural_network,
                      LE_LAYER(le_activation_layer_new("A1", LE_ACTIVATION_SIGMOID)));
    /// @note: Single Layer Perceptron is unable to solve XOR problem / non-linearly-separable
    le_sequential_add(neural_network,
                      LE_LAYER(le_dense_layer_new("D2", 2, 1)));
    le_sequential_add(neural_network,
                      LE_LAYER(le_activation_layer_new("A2", LE_ACTIVATION_SIGMOID)));

    le_sequential_set_loss(neural_network, LE_LOSS_LOGISTIC);

    LE_INFO("Initial Gradcheck. Average Normalized Euclidian Distance = %f", le_sequential_check_gradients(neural_network, x, y, 1e-2f));

    le_sequential_to_dot(neural_network, "snn.dot");
    
    LE_INFO("Training Neural Network");
    LeBGD *optimizer = le_bgd_new(LE_MODEL(neural_network), x, y, 1.f);
    for (unsigned i = 0; i <= 1000; i++)
    {
        le_optimizer_step(LE_OPTIMIZER(optimizer));

        if ((i % 100) == 0)
        {
            LE_INFO("Iteration %u", i);
            LeTensor *h = le_model_predict(LE_MODEL(neural_network), x);
            LE_INFO("Training Error = %f", le_logistic_loss(h, y));
            le_tensor_unref(h);
            LE_INFO("Gradcheck. Average Normalized Euclidian Distance = %f", le_sequential_check_gradients(neural_network, x, y, 1e-2f));
        }
    }
    
    g_object_unref(optimizer);
    
    LeTensor *h = le_model_predict(LE_MODEL(neural_network), x);
    printf("Predicted value =\n");
    le_tensor_print(h, stdout);
    
    le_tensor_unref(h);
    le_tensor_unref(y);
    le_tensor_unref(x);
    
    return 0;
}

