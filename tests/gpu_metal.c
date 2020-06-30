#include <stdlib.h>
#include <le/le.h>

void le_metal_init(void);

LeTensor * le_tensor_to_metal(const LeTensor *another);

int main(int argc, char *argv[])
{
    le_metal_init();
    
    LeTensor *cx = le_tensor_new(LE_TYPE_FLOAT32, 2, 3, 3,   
        1.0, 2.0, 1.0,
        0.0, 0.0, 0.0,
        -1.0, -2.0, -1.0
    );

    LeTensor *cy = le_tensor_new(LE_TYPE_FLOAT32, 2, 3, 3,   
        1.0, 0.0, -1.0,
        2.0, 0.0, -2.0,
        1.0, 0.0, -1.0
    );

    LeTensor *cmul = le_matrix_new_product(cx, cy);
    le_tensor_print(cmul, stdout);

    LeTensor *mx = le_tensor_to_metal(cx);
    LeTensor *my = le_tensor_to_metal(cx);

    LeTensor *gmul = le_matrix_new_product(mx, my);
    le_tensor_print(gmul, stdout);

    le_tensor_free(my);
    le_tensor_free(mx);

    le_tensor_free(cmul);

    le_tensor_free(cy);
    le_tensor_free(cx);

    return EXIT_SUCCESS;
}
