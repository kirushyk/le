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
        a = le_matrix_new_identity(LE_TYPE_FLOAT32, height);
        assert(le_test_ensure_matrix_size(a, height, height));
        le_tensor_free(a);
        
        for (width = 1; width < MAX_DIMENSION; width++)
        {
            a = le_matrix_new_zeros(LE_TYPE_FLOAT32, height, width);
            assert(le_test_ensure_matrix_size(a, height, width));
            le_tensor_free(a);
            
            a = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, height, width);
            assert(le_test_ensure_matrix_size(a, height, width));
            
            for (second_width = 1; second_width < MAX_DIMENSION; second_width++)
            {
                b = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, width, second_width);
                c = le_matrix_new_product(a, b);
                assert(le_test_ensure_matrix_size(c, height, second_width));
                le_tensor_free(c);
                le_tensor_free(b);
            }
            
            le_tensor_free(a);
        }
    }

    a = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, 10, 5);
    b = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, 10, 5);
    LeTensor *at = le_matrix_new_transpose(a);
    c = le_matrix_new_product(at, b);
    LeTensor *d = le_matrix_new_product_full(a, true, b, false);
    printf("c = ");
    le_tensor_print(c, stdout);
    printf("d = ");
    le_tensor_print(d, stdout);
    assert(le_tensor_equal(c, d));
    le_tensor_free(d);
    le_tensor_free(c);
    le_tensor_free(at);
    LeTensor *bt = le_matrix_new_transpose(b);
    c = le_matrix_new_product(a, bt);
    d = le_matrix_new_product_full(a, false, b, true);
    float sad = le_tensor_sad_f32(c, d);
    assert(sad < 1e-3f);
    le_tensor_free(bt);
    le_tensor_free(b);
    le_tensor_free(a);
    
    return EXIT_SUCCESS;
}
