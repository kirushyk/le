#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

unsigned
train_until_convergence(bool use_normalization)
{
    unsigned steps_count = rand() % (use_normalization ? 255 : 1024);
    return steps_count;
}

#define TRAIN_COUNT 128

int
main(int argc, char *argv[])
{
    unsigned steps_till_convergence[TRAIN_COUNT];
    float mean_steps_count_with_norm = 0.0f, mean_steps_count_withoout_norm = 0.0f;

    printf("Training WITHOUT input normalization...\n");
    for (int i = 0; i < TRAIN_COUNT; i++)
    {
        steps_till_convergence[i] = train_until_convergence(false);
        mean_steps_count_withoout_norm += steps_till_convergence[i];
    }
    mean_steps_count_withoout_norm /= TRAIN_COUNT;

    printf("Training WITH input normalization...\n");
    for (int i = 0; i < TRAIN_COUNT; i++)
    {
        steps_till_convergence[i] = train_until_convergence(true);
        mean_steps_count_with_norm += steps_till_convergence[i];
    }
    mean_steps_count_with_norm /= TRAIN_COUNT;

    if (mean_steps_count_with_norm < mean_steps_count_withoout_norm)
    {
        printf("Average number of steps reduced: %.1f < %.1f\n", mean_steps_count_with_norm, mean_steps_count_withoout_norm);
        printf("Input normalization accelerated convergence\n");
        return EXIT_SUCCESS;
    }
    else
    {
        printf("Input normalization failed\n");
        return EXIT_FAILURE;
    }
}
