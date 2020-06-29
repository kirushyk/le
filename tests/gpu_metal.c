#include <stdlib.h>
#include <le/le.h>

void le_metal_init(void);

int main(int argc, char *argv[])
{
    le_metal_init();
    
    LeTensor *mx1 = le_tensor_new(LE_TYPE_FLOAT32, 2, 3, 3,   
        1.0, 2.0, 1.0,
        0.0, 0.0, 0.0,
        -1.0, -2.0, -1.0
    );

    le_tensor_free(mx1);

    return EXIT_SUCCESS;
}
