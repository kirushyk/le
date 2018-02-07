#include <stdlib.h>
#include <le/le.h>

#define MIN_DIMENSION 16
#define MAX_DIMENSION 48

int
main()
{
    unsigned width;
    unsigned height;
    unsigned second_width;
    
    LeMatrix *a;
    LeMatrix *b;
    LeMatrix *c;
    
    for (height = MIN_DIMENSION; height <= MAX_DIMENSION; height++)
    {
        for (width = MIN_DIMENSION; width <= MAX_DIMENSION; width++)
        {
            a = le_matrix_new_rand(height, width);
            
            for (second_width = MIN_DIMENSION; second_width <= MAX_DIMENSION; second_width++)
            {
                b = le_matrix_new_rand(width, second_width);
                c = le_matrix_new_product(a, b);
                le_matrix_free(c);
                le_matrix_free(b);
            }
            
            le_matrix_free(a);
        }
    }
    return EXIT_SUCCESS;
}
