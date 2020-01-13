/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

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
        -1.0f, 1.0f, 1.0f, -1.0f
    };
    LeTensor *y = le_matrix_new_from_data(1, 4, y_data);
    
    printf("Train set:\n");
    printf("x =\n");
    le_tensor_print(x, stdout);
    printf("y =\n");
    le_tensor_print(y, stdout);

    LeSequential *neural_network = le_sequential_new();
    le_sequential_add(neural_network, LE_LAYER(le_dense_layer_new(2, 4)));
    le_sequential_add(neural_network, LE_LAYER(le_activation_layer_new(LE_ACTIVATION_TANH)));
    le_sequential_add(neural_network, LE_LAYER(le_dense_layer_new(4, 1)));
    le_sequential_add(neural_network, LE_LAYER(le_activation_layer_new(LE_ACTIVATION_TANH)));

    LeTensor *h = le_model_predict((LeModel *)neural_network, x);
    printf("Predicted value =\n");
    le_tensor_print(h, stdout);
    
    le_tensor_free(h);
    le_tensor_free(y);
    le_tensor_free(x);
    
    return 0;
}

