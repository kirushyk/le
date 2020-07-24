#include <assert.h>
#include <stdlib.h>
#include <le/le.h>
#include <platform/metal/lemetal.h>

int main(int argc, char *argv[])
{
    le_metal_init();
    
    LeTensor *a = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, 8, 8);
    LeTensor *b = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, 8, 8);
    
    LeTensor *a_copy = le_tensor_new_copy(a);
    le_tensor_mul(a_copy, b);
    le_tensor_print(a_copy, stdout);

    LeTensor *ma = le_tensor_to_metal(a);
    LeTensor *mb = le_tensor_to_metal(b);
    
    LeTensor *ma_copy = le_tensor_new_copy(ma);
    le_tensor_mul(ma_copy, mb);
    LeTensor *result = le_tensor_to_cpu(ma_copy);
    le_tensor_print(result, stdout);
    
    le_tensor_sub(result, a_copy);
    float l2 = le_tensor_l2_f32(result);
    assert(l2 < 1e-4f);
    
    le_tensor_free(result);
    le_tensor_free(ma_copy);
    le_tensor_free(ma);
    le_tensor_free(mb);
    le_tensor_free(a_copy);
    le_tensor_free(a);
    le_tensor_free(b);

    return EXIT_SUCCESS;
}
