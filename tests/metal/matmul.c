#include <assert.h>
#include <stdlib.h>
#include <le/le.h>
#include <backends/metal/lemetal.h>

int main(int argc, char *argv[])
{
    le_metal_init();
    
    LeTensor *ci = le_matrix_new_identity(LE_TYPE_F32, 5);
    LeTensor *gi = le_tensor_to_metal(ci);
    LeTensor *ri = le_tensor_to_cpu(gi);
    le_tensor_print(ri, stdout);
    assert(le_tensor_equal(ci, ri));
    le_tensor_unref(ri);
    le_tensor_unref(gi);
    le_tensor_unref(ci);
    
    LeTensor *cx = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, 8, 8);

    LeTensor *cy = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, 8, 8);

    LeTensor *cmul = le_matrix_new_product(cx, cy);
    le_tensor_print(cmul, stdout);

    LeTensor *mx = le_tensor_to_metal(cx);
    LeTensor *my = le_tensor_to_metal(cy);

    LeTensor *gmul = le_matrix_new_product(mx, my);
    LeTensor *result = le_tensor_to_cpu(gmul);
    le_tensor_print(result, stdout);
    
    le_tensor_sub(result, cmul);
    gfloat l2 = le_tensor_l2_f32(result);
    assert(l2 < 1e-4f);
    
    le_tensor_unref(result);
    le_tensor_unref(gmul);

    le_tensor_unref(my);
    le_tensor_unref(mx);

    le_tensor_unref(cmul);

    le_tensor_unref(cy);
    le_tensor_unref(cx);

    return EXIT_SUCCESS;
}
