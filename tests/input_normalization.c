#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <le/le.h>

/** @note: This function will return amount of gradient descent steps taken to
 *         fit shallow neural network to dataset scaled in different dimensions
 */
unsigned
train_until_convergence(bool use_normalization)
{
    unsigned steps_count = 0;

    LeTensor *x = le_tensor_new(LE_TYPE_FLOAT32, 2, 2, 4,
        1.0, 2.0, 3.0, 4.0,
        400.0, 300.0, 200.0, 100.0
    );
    
    LeTensor *y = le_tensor_new(LE_TYPE_FLOAT32, 2, 1, 4,
        0.0, 0.0, 1.0, 1.0
    );
    
    steps_count = rand() % (use_normalization ? 255 : 1024);

    le_tensor_free(y);
    le_tensor_free(x);

    return steps_count;
}

#define TRAIN_COUNT 100

/** @note: Idea of this test is to check whether normalization will accelerate
 *         gradient descent convergence in stretched dataset
 */
int
main(int argc, char *argv[])
{
    unsigned steps_till_convergence[TRAIN_COUNT];
    float mean_steps_count_with_norm = 0.0f, mean_steps_count_withoout_norm = 0.0f;

    printf("Training WITHOUT input normalization...\n");
    for (int i = 0; i < TRAIN_COUNT; i++)
    {
        printf("\33[2K\r%d / %d", i + 1, TRAIN_COUNT);
        steps_till_convergence[i] = train_until_convergence(false);
        mean_steps_count_withoout_norm += steps_till_convergence[i];
    }
    mean_steps_count_withoout_norm /= TRAIN_COUNT;

    printf("\nTraining WITH input normalization...\n");
    for (int i = 0; i < TRAIN_COUNT; i++)
    {
        printf("\33[2K\r%d / %d", i + 1, TRAIN_COUNT);
        steps_till_convergence[i] = train_until_convergence(true);
        mean_steps_count_with_norm += steps_till_convergence[i];
    }
    mean_steps_count_with_norm /= TRAIN_COUNT;

    if (mean_steps_count_with_norm < mean_steps_count_withoout_norm)
    {
        printf("\nAverage number of steps reduced: %.1f < %.1f\n", mean_steps_count_with_norm, mean_steps_count_withoout_norm);
        printf("Input normalization accelerated convergence\n");
        return EXIT_SUCCESS;
    }
    else
    {
        printf("Input normalization failed\n");
        return EXIT_FAILURE;
    }
}
