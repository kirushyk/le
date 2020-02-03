/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "snn"

#include <stdio.h>
#include <math.h>
#include <le/le.h>

int
main(int argc, const char *argv[])
{
    const float x_data[] =
    {
        1.0f, 2.0f, 1.0f, 2.0f,
        2.0f, 2.0f, 1.0f, 1.0f
    };
    LeTensor *x = le_matrix_new_from_data(2, 4, x_data);
    
    const float y_data[] =
    {
        0.0f, 1.0f, 1.0f, 0.0f
    };
    LeTensor *y = le_matrix_new_from_data(1, 4, y_data);
    
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
    le_sequential_add(neural_network,
                      LE_LAYER(le_dense_layer_new("D2", 2, 1)));
    le_sequential_add(neural_network,
                      LE_LAYER(le_activation_layer_new("A2", LE_ACTIVATION_SIGMOID)));
    
    LE_INFO("Training Neural Network");
    LeBGD *optimizer = le_bgd_new(le_model_get_parameters(LE_MODEL(neural_network)),
                                  3.f);
    for (unsigned i = 0; i <= 1000; i++)
    {

        LeList *gradients = le_model_get_gradients(LE_MODEL(neural_network),
                                                   x, y);
        LE_OPTIMIZER(optimizer)->gradients = gradients;
        le_optimizer_step(LE_OPTIMIZER(optimizer));

        if ((i % 100) == 0)
        {
            LE_INFO("Iteration %u", i);
            LeTensor *h = le_model_predict(LE_MODEL(neural_network), x);
            LE_INFO("Training Error = %f", le_cross_entropy(h, y));
            le_tensor_free(h);
        }

        le_list_foreach(gradients, (LeFunction)le_tensor_free);
    }
    
    le_bgd_free(optimizer);
    
    LeTensor *h = le_model_predict(LE_MODEL(neural_network), x);
    printf("Predicted value =\n");
    le_tensor_print(h, stdout);
    
    le_tensor_free(h);
    le_tensor_free(y);
    le_tensor_free(x);
    
    return 0;
}

