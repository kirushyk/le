#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <le/le.h>

#define DEFAULT_LOG_CATEGORY "gradcheck"

int
main(int argc, char *argv[])
{
    LeSequential *nn = le_sequential_new();
    le_sequential_add(nn, LE_LAYER(le_dense_layer_new("FC1", 2, 1)));
    le_sequential_add(nn, LE_LAYER(le_activation_layer_new("A1", LE_ACTIVATION_SIGMOID)));
    le_sequential_set_loss(nn, LE_LOSS_LOGISTIC);

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
    LeList *gradients_iterator, *gradients_estimations_iterator;
    bool mismatch_found = false;
    for (gradients_iterator = gradients, gradients_estimations_iterator = gradients_estimations;
         gradients_iterator && gradients_estimations_iterator;
         gradients_iterator = gradients_iterator->next, gradients_estimations_iterator = gradients_estimations_iterator->next)
    {
        LeTensor *gradient_estimate = (LeTensor *)gradients_estimations_iterator->data;
        printf("gradient_estimate = \n");
        le_tensor_print(gradient_estimate, stdout);
        LeTensor *gradient = (LeTensor *)gradients_iterator->data;
        printf("gradient = \n");
        le_tensor_print(gradient, stdout);
        float denominator = le_tensor_l2_f32(gradient) + le_tensor_l2_f32(gradient_estimate);
        if (denominator > 0.0f)
        {
            le_tensor_sub(gradient_estimate, gradient);
            float normalized_distance = le_tensor_l2_f32(gradient_estimate) / denominator;
            LE_INFO("Normalized distance between gradient estimation and actual gradient: %f", normalized_distance);
            if (normalized_distance > 1e-7f)
            {
                LE_ERROR("Normalized distance too large: %f", normalized_distance);
                mismatch_found = true;
            }
        }
    }
    if (gradients_iterator)
    {
        LE_ERROR("Some gradients estimations missing or extra gradients present");
        return EXIT_FAILURE;
    }
    if (gradients_estimations_iterator)
    {
        LE_ERROR("Some gradients missing or extra gradients estimations present");
        return EXIT_FAILURE;
    }
    if (mismatch_found)
    {
        return EXIT_FAILURE;
    }
    le_list_foreach(gradients_estimations, (LeFunction)le_tensor_free);
    le_list_foreach(gradients, (LeFunction)le_tensor_free);
    le_bgd_free(optimizer);
    le_tensor_free(y);
    le_tensor_free(x);

    le_sequential_free(nn);
        
    return EXIT_SUCCESS;
}
