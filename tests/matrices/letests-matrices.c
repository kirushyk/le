#include "letests-matrices.h"
#include <le/le.h>

int
le_test_ensure_matrix_size(LeMatrix *a, unsigned height, unsigned width)
{
    if (a == NULL)
    {
        printf("NULL pointer given");
        return 1;
    }
    
    if (le_matrix_get_width(a) != width)
    {
        printf("Wrong matrix width.\n");
        return 1;
    }
    
    if (le_matrix_get_height(a) != height)
    {
        printf("Wrong matrix height.\n");
        return 1;
    }
    
    return 0;
}

int
le_test_matrices()
{
    unsigned width;
    unsigned height;
    LeMatrix *a;
    
    for (height = 1; height < 4; height++)
    {
        a = le_matrix_new_identity(height);
        if (le_test_ensure_matrix_size(a, height, height) != 0)
        {
            printf("Problem occured when testing sizes of identity matrix.\n");
            return 0;
        }
        le_matrix_free(a);
        
        for (width = 1; width < 4; width++)
        {
            a = le_matrix_new_zeros(height, width);
            if (le_test_ensure_matrix_size(a, height, width) != 0)
            {
                printf("Problem occured when testing sizes of zeros matrix.\n");
                return 0;
            }
            le_matrix_free(a);
            
            a = le_matrix_new_rand(height, width);
            if (le_test_ensure_matrix_size(a, height, width) != 0)
            {
                printf("Problem occured when testing sizes of random matrix.\n");
                return 0;
            }
            le_matrix_free(a);
        }
    }
    return 0;
}
