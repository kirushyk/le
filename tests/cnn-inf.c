#define DEFAULT_LOG_CATEGORY "cnn-inf"

#include <assert.h>
#include <stdlib.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>

int
main (int argc, char *argv[])
{
  LeSequential *nn = le_sequential_new ();
  le_sequential_add (nn, LE_LAYER (le_conv2d_new ("C1", 3, 1, 1, 0, 1)));
  le_sequential_free (nn);

  LeTensor *input = le_tensor_new (LE_TYPE_F32, 4, 1, 6, 6, 1, // clang-format off
      1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
      1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
      1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
      0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
      0.0, 0.0, 0.0, 1.0, 1.0, 1.0
  ); // clang-format on
  LeTensor *expected_output = le_tensor_new (LE_TYPE_F32, 4, 1, 4, 4, 1, // clang-format off
      0.0, 0.0, 0.0, 0.0,
      4.0, 2.0, -2.0, -4.0,
      4.0, 2.0, -2.0, -4.0,
      0.0, 0.0, 0.0, 0.0
  ); // clang-format on
  LeTensor *output = le_model_predict (LE_MODEL (nn), input);
  LeShape *expected_shape = le_shape_new (4, 1, 4, 4, 1);
  assert (le_shape_equal (output->shape, expected_shape));
  le_shape_unref (expected_shape);
  le_tensor_unref (output);
  le_tensor_unref (input);

  return EXIT_SUCCESS;
}
