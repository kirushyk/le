#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#ifndef __APPLE__
#   include <time.h>
#endif
#include <le/le.h>

#define DEFAULT_LOG_CATEGORY "gradcheck"

int
main(int argc, char *argv[])
{
#ifdef __APPLE__
    sranddev();
#else
    srand(time(NULL));
#endif

    const gfloat epsilon = 1e-3f;
    LeTensor *x = le_tensor_new(LE_TYPE_FLOAT32, 2, 2, 4,
        1.0, 2.0, 3.0, 4.0,
        4.0, 3.0, 2.0, 1.0
    );
    LeTensor *y = le_tensor_new(LE_TYPE_FLOAT32, 2, 1, 4,
        0.0, 0.0, 1.0, 1.0
    );

    LE_INFO("Initializing...");
    LeSequential *nn = le_sequential_new();
    le_sequential_add(nn, LE_LAYER(le_dense_layer_new("FC1", 2, 1)));
    le_sequential_add(nn, LE_LAYER(le_activation_layer_new("A1", LE_ACTIVATION_SIGMOID)));
    le_sequential_set_loss(nn, LE_LOSS_LOGISTIC);
    LE_INFO("cost = %f", le_sequential_compute_cost(nn, x, y));
    gfloat average_normalized_distance = le_sequential_check_gradients(nn, x, y, epsilon);
    LE_INFO("average normalized distance = %f", average_normalized_distance);
    bool failed = (average_normalized_distance > epsilon);

    LE_INFO("Pretraining...");
    LeBGD *optimizer = le_bgd_new(LE_MODEL(nn), x, y, 1.0f);
    for (unsigned i = 0; i <= 5; i++)
    {
        le_optimizer_step(LE_OPTIMIZER(optimizer));
    }  
    LE_INFO("cost = %f", le_sequential_compute_cost(nn, x, y));
    average_normalized_distance = le_sequential_check_gradients(nn, x, y, epsilon);
    LE_INFO("average normalized distance = %f", average_normalized_distance);
    failed |= (average_normalized_distance > epsilon);

    LE_INFO("Training...");
    for (unsigned i = 0; i <= 50; i++)
    {
        le_optimizer_step(LE_OPTIMIZER(optimizer));
    }
    g_object_unref(optimizer);
    LE_INFO("cost = %f", le_sequential_compute_cost(nn, x, y));
    average_normalized_distance = le_sequential_check_gradients(nn, x, y, epsilon);
    LE_INFO("average normalized distance = %f", average_normalized_distance);
    failed |= (average_normalized_distance > epsilon);
    g_object_unref(nn);

    le_tensor_free(y);
    le_tensor_free(x);
        
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
