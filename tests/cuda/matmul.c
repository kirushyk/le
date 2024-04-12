#include <assert.h>
#include <stdlib.h>
#include <le/le.h>
#include <backends/cuda/lecuda.h>

int main(int argc, char *argv[])
{    
    LeTensor *ci = le_matrix_new_identity(LE_TYPE_FLOAT32, 5);
    LeTensor *gi = le_tensor_to_cuda(ci);
    LeTensor *ri = le_cuda_tensor_to_cpu(gi);
    le_tensor_print(ri, stdout);
    assert(le_tensor_equal(ci, ri));
    le_tensor_free(ri);
    le_tensor_free(gi);
    le_tensor_free(ci);
    
    LeTensor *cx = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, 8, 8);

    LeTensor *cy = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, 8, 8);

    LeTensor *cmul = le_matrix_new_product(cx, cy);
    le_tensor_print(cmul, stdout);

    LeTensor *mx = le_tensor_to_cuda(cx);
    LeTensor *my = le_tensor_to_cuda(cy);

    LeTensor *gmul = le_matrix_new_product(mx, my);
    LeTensor *result = le_cuda_tensor_to_cpu(gmul);
    le_tensor_print(result, stdout);
    
    le_tensor_sub(result, cmul);
    gfloat l2 = le_tensor_l2_f32(result);
    assert(l2 < 1e-4f);
    
    le_tensor_free(result);
    le_tensor_free(gmul);

    le_tensor_free(my);
    le_tensor_free(mx);

    le_tensor_free(cmul);

    le_tensor_free(cy);
    le_tensor_free(cx);

    return EXIT_SUCCESS;
}
