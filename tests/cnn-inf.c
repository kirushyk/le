#define DEFAULT_LOG_CATEGORY "cnn-inf"

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>

int
main(int argc, char *argv[])
{
    LeSequential *nn = le_sequential_new();
    le_sequential_add(nn, LE_LAYER(le_conv2d_new("C1", 3, 1, 1, 0, 1)));
    le_sequential_free(nn);

    LeTensor *input = le_tensor_new(LE_TYPE_FLOAT32, 4, 1, 1, 6, 6,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0
    );
    LeTensor *output = le_model_predict(LE_MODEL(nn), input);
    le_tensor_free(output);
    le_tensor_free(input);

    return EXIT_SUCCESS;
}
