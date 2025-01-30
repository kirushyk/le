#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <le/le.h>

#define MAX_STEPS 1000

/** @note: This function will return amount of gradient descent steps taken to
 *         fit shallow neural network to dataset scaled in different dimensions
 */
unsigned
train_until_convergence (bool use_normalization)
{
  unsigned steps_count = MAX_STEPS;

  LeTensor *x = use_normalization ? le_tensor_new (LE_TYPE_F32, 2, 2, 4, // clang-format off
          1.0, 2.0, 3.0, 4.0,
          4.0, 3.0, 2.0, 1.0
      ) // clang-format on
                                  : le_tensor_new (LE_TYPE_F32, 2, 2, 4, // clang-format off
          1.0, 2.0, 3.0, 4.0,
          40.0, 30.0, 20.0, 10.0
      ); // clang-format on

  LeTensor *y = le_tensor_new (LE_TYPE_F32, 2, 1, 4, // clang-format off
      0.0, 0.0, 1.0, 1.0
  ); // clang-format on

  LeSequential *nn = le_sequential_new ();
  le_sequential_add (nn, LE_LAYER (le_dense_layer_new ("FC1", 2, 1)));
  le_sequential_add (nn, LE_LAYER (le_activation_layer_new ("A1", LE_ACTIVATION_SIGMOID)));
  LeBGD *optimizer = le_bgd_new (LE_MODEL (nn), x, y, 1.0f);
  for (unsigned i = 0; i <= MAX_STEPS; i++) {
    le_optimizer_step (LE_OPTIMIZER (optimizer));
    LeTensor *h = le_model_predict (LE_MODEL (nn), x);
    gfloat loss = le_logistic_loss (h, y);
    le_tensor_unref (h);
    if (loss < 1.0f) {
      steps_count = i;
      break;
    }
  }

  g_object_unref (optimizer);
  g_object_unref (nn);

  le_tensor_unref (y);
  le_tensor_unref (x);

  return steps_count;
}

#define TRAIN_COUNT 100

/** @note: Idea of this test is to check whether normalization will accelerate
 *         gradient descent convergence in stretched dataset
 */
int
main (int argc, char *argv[])
{
  unsigned steps_till_convergence[TRAIN_COUNT];
  gfloat mean_steps_count_with_norm = 0.0f, mean_steps_count_without_norm = 0.0f;

  printf ("Training WITHOUT input normalization...\n");
  for (int i = 0; i < TRAIN_COUNT; i++) {
    printf ("\33[2K\r%d / %d", i + 1, TRAIN_COUNT);
    fflush (stdout);
    steps_till_convergence[i] = train_until_convergence (false);
    mean_steps_count_without_norm += steps_till_convergence[i];
  }
  mean_steps_count_without_norm /= TRAIN_COUNT;
  printf ("\nAverage number of steps taken: %.1f\n", mean_steps_count_without_norm);

  printf ("Training WITH input normalization...\n");
  for (int i = 0; i < TRAIN_COUNT; i++) {
    printf ("\33[2K\r%d / %d", i + 1, TRAIN_COUNT);
    fflush (stdout);
    steps_till_convergence[i] = train_until_convergence (true);
    mean_steps_count_with_norm += steps_till_convergence[i];
  }
  mean_steps_count_with_norm /= TRAIN_COUNT;
  printf ("\nAverage number of steps taken: %.1f\n", mean_steps_count_with_norm);

  if (mean_steps_count_with_norm < mean_steps_count_without_norm) {
    printf (
        "Average number of steps reduced: %.1f < %.1f\n", mean_steps_count_with_norm, mean_steps_count_without_norm);
    printf ("Input normalization accelerated convergence\n");
    return EXIT_SUCCESS;
  } else {
    printf ("Input normalization failed\n");
    return EXIT_FAILURE;
  }
}
