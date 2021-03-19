#define DEFAULT_LOG_CATEGORY "cnn-inf"

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>

int
main(int argc, char *argv[])
{
    LeSequential *nn = le_sequential_new();
    le_sequential_add(nn, LE_LAYER(le_conv2d_new("C1", 3, 1, 1, 0, 1)));
    le_sequential_free(nn);

    LeTensor *input = le_tensor_new(LE_TYPE_FLOAT32, 4, 1, 6, 6, 1,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0
    );
    LeTensor *expected_output = le_tensor_new(LE_TYPE_FLOAT32, 4, 1, 4, 4, 1,
        0.0, 0.0, 0.0, 0.0,
        4.0, 2.0, -2.0, -4.0,
        4.0, 2.0, -2.0, -4.0,
        0.0, 0.0, 0.0, 0.0
    );
    LeTensor *output = le_model_predict(LE_MODEL(nn), input);
    LeShape *expected_shape = le_shape_new(4, 1, 4, 4, 1);
    assert(le_shape_equal(output->shape, expected_shape));
    le_shape_free(expected_shape);
    le_tensor_free(output);
    le_tensor_free(input);

    return EXIT_SUCCESS;
}
