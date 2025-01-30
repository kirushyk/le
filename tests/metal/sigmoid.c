#include <assert.h>
#include <stdlib.h>
#include <le/le.h>
#include <backends/metal/lemetal.h>

int main(int argc, char *argv[])
{   
    le_metal_init();

    LeTensor *a = le_tensor_new(LE_TYPE_F32, 2, 4, 4,
        0.0, 0.1, 0.2, 0.3,
        0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 1.0, 1.1,
        1.2, 1.3, 1.4, 1.5
    );
    LeTensor *ca = le_tensor_to_metal(a);
    le_tensor_apply_sigmoid(a);
    le_tensor_apply_sigmoid(ca);
    LeTensor *result = le_tensor_to_cpu(ca);
    // le_tensor_print(result, stdout);
    // le_tensor_print(a, stdout);

    le_tensor_sub(result, a);
    gfloat l2 = le_tensor_l2_f32(result);
    assert(l2 < 1e-4f);

    le_tensor_unref(result);
    le_tensor_unref(ca);
    le_tensor_unref(a);

    a = le_tensor_new(LE_TYPE_F32, 2, 4, 4,
        0.0, 0.1, 0.2, 0.3,
        0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 1.0, 1.1,
        1.2, 1.3, 1.4, 1.5
    );
    ca = le_tensor_to_metal(a);
    le_tensor_apply_sigmoid_prime(a);
    le_tensor_apply_sigmoid_prime(ca);
    result = le_tensor_to_cpu(ca);
    // le_tensor_print(result, stdout);
    // le_tensor_print(a, stdout);

    le_tensor_sub(result, a);
    l2 = le_tensor_l2_f32(result);
    assert(l2 < 1e-4f && "Sigmoid Prime on Metal");

    le_tensor_unref(result);
    le_tensor_unref(ca);
    le_tensor_unref(a);

    return EXIT_SUCCESS;
}
