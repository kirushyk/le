#include <assert.h>
#include <stdlib.h>
#include <le/le.h>
#include <platform/cuda/lecuda.h>

int main(int argc, char *argv[])
{   
    LeTensor *a = le_tensor_new(LE_TYPE_FLOAT32, 2, 4, 4,
        0.0, 0.1, 0.2, 0.3,
        0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 1.0, 1.1,
        1.2, 1.3, 1.4, 1.5
    );
    LeTensor *b = le_tensor_new(LE_TYPE_FLOAT32, 2, 4, 4,
        1.0, 1.1, 1.0, 1.1,
        1.1, 1.0, 1.1, 1.0,
        1.0, 1.1, 1.0, 1.1,
        1.1, 1.0, 1.1, 1.0
    );
    
    LeTensor *a_copy = le_tensor_new_copy(a);
    le_tensor_mul(a_copy, b);
    le_tensor_print(a_copy, stdout);

    LeTensor *ma = le_tensor_to_cuda(a);
    LeTensor *mb = le_tensor_to_cuda(b);
    
    LeTensor *ma_copy = le_tensor_new_copy(ma);
    le_tensor_mul(ma_copy, mb);
    LeTensor *result = le_cuda_tensor_to_cpu(ma_copy);
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
