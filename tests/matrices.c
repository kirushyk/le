/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>

int
le_test_ensure_matrix_size(LeTensor *a, unsigned height, unsigned width)
{
    if (a == NULL)
    {
        fprintf(stderr, "NULL pointer given");
        return 0;
    }
    
    if (le_matrix_get_width(a) != width)
    {
        fprintf(stderr, "Wrong matrix width.\n");
        return 0;
    }
    
    if (le_matrix_get_height(a) != height)
    {
        fprintf(stderr, "Wrong matrix height.\n");
        return 0;
    }
    
    return 1;
}

#define MAX_DIMENSION 4

int
main()
{
    unsigned width;
    unsigned height;
    unsigned second_width;
    
    LeTensor *a;
    LeTensor *b;
    LeTensor *c;
    
    for (height = 1; height < MAX_DIMENSION; height++)
    {
        a = le_matrix_new_identity(height);
        assert(le_test_ensure_matrix_size(a, height, height));
        le_tensor_free(a);
        
        for (width = 1; width < MAX_DIMENSION; width++)
        {
            a = le_matrix_new_zeros(height, width);
            assert(le_test_ensure_matrix_size(a, height, width));
            le_tensor_free(a);
            
            a = le_matrix_new_rand(height, width);
            assert(le_test_ensure_matrix_size(a, height, width));
            
            for (second_width = 1; second_width < MAX_DIMENSION; second_width++)
            {
                b = le_matrix_new_rand(width, second_width);
                c = le_matrix_new_product(a, b);
                assert(le_test_ensure_matrix_size(c, height, second_width));
                le_tensor_free(c);
                le_tensor_free(b);
            }
            
            le_tensor_free(a);
        }
    }
    return EXIT_SUCCESS;
}
