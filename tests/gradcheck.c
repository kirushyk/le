#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <le/le.h>

int
main(int argc, char *argv[])
{
    LeSequential *nn = le_sequential_new();
    le_sequential_add(nn, LE_LAYER(le_dense_layer_new("FC1", 2, 1)));
    le_sequential_add(nn, LE_LAYER(le_activation_layer_new("A1", LE_ACTIVATION_SIGMOID)));

    printf("Pretraining...\n");
    LeTensor *x = le_tensor_new(LE_TYPE_FLOAT32, 2, 2, 4,
        1.0, 2.0, 3.0, 4.0,
        4.0, 3.0, 2.0, 1.0
    );
    LeTensor *y = le_tensor_new(LE_TYPE_FLOAT32, 2, 1, 4,
        0.0, 0.0, 1.0, 1.0
    );
    LeBGD *optimizer = le_bgd_new(LE_MODEL(nn), x, y, 1.0f);
    for (unsigned i = 0; i <= 10; i++)
    {
        le_optimizer_step(LE_OPTIMIZER(optimizer));
    }
    LeList *gradients = le_model_get_gradients(LE_MODEL(nn), x, y);
    LeList *gradients_estimations = le_sequential_estimate_gradients(nn, x, y);
    le_list_foreach(gradients_estimations, (LeFunction)le_tensor_free);
    le_list_foreach(gradients, (LeFunction)le_tensor_free);
    le_bgd_free(optimizer);
    le_tensor_free(y);
    le_tensor_free(x);

    le_sequential_free(nn);
        
    return EXIT_SUCCESS;
}
