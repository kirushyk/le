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
    
    LeTensor *a, *at, *b, *bt, *c, *d, *ab, *ab_check, *ba, *ba_check;

    a = le_tensor_new(LE_TYPE_FLOAT32, 2, 3, 2,
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    );

    b = le_tensor_new(LE_TYPE_FLOAT32, 2, 2, 3,
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    );

    ab_check = le_tensor_new(LE_TYPE_FLOAT32, 2, 3, 3,
        9.0,  12.0, 15.0,
        19.0, 26.0, 33.0,
        29.0, 40.0, 51.0
    );

    ab = le_matrix_new_product(a, b);
    printf("ab = ");
    le_tensor_print(ab, stdout);
    printf("ab_check = ");
    le_tensor_print(ab_check, stdout);
    assert(le_tensor_equal(ab, ab_check));
    le_tensor_free(ab);

    bt = le_matrix_new_transpose(b);
    ab = le_matrix_new_product_full(a, false, bt, true);
    printf("ab (of t) = ");
    le_tensor_print(ab, stdout);
    printf("ab_check = ");
    le_tensor_print(ab_check, stdout);
    printf ("sad afbt %f\n", le_tensor_sad_f32(ab, ab_check));
    assert(le_tensor_sad_f32(ab, ab_check) < 1e-3f);
    le_tensor_free(ab);
    le_tensor_free(bt);

    at = le_matrix_new_transpose(a);
    ab = le_matrix_new_product_full(at, true, b, false);
    printf("ab (of t) = ");
    le_tensor_print(ab, stdout);
    printf("ab_check = ");
    le_tensor_print(ab_check, stdout);
    printf ("sad atbf %f\n", le_tensor_sad_f32(ab, ab_check));
    assert(le_tensor_sad_f32(ab, ab_check) < 1e-3f);
    le_tensor_free(ab);
    le_tensor_free(at);

    le_tensor_free(ab_check);

    ba = le_matrix_new_product(b, a);
    ba_check = le_tensor_new(LE_TYPE_FLOAT32, 2, 2, 2,
        22.0, 28.0,
        49.0, 64.0
    );
    assert(le_tensor_equal(ba, ba_check));
    le_tensor_free(ba_check);
    le_tensor_free(ba);

    le_tensor_free(b);
    le_tensor_free(a);

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
    at = le_matrix_new_transpose(a);
    c = le_matrix_new_product(at, b);
    d = le_matrix_new_product_full(a, true, b, false);
    printf("c = ");
    le_tensor_print(c, stdout);
    printf("d = ");
    le_tensor_print(d, stdout);
    printf ("sad atbf %f\n", le_tensor_sad_f32(c, d));
    assert(le_tensor_sad_f32(c, d) < 1e-3f);
    le_tensor_free(d);
    le_tensor_free(c);
    le_tensor_free(at);
    bt = le_matrix_new_transpose(b);
    c = le_matrix_new_product(a, bt);
    d = le_matrix_new_product_full(a, false, b, true);
    printf ("sad afbt %f\n", le_tensor_sad_f32(c, d));
    assert(le_tensor_sad_f32(c, d) < 1e-3f);
    le_tensor_free(bt);
    le_tensor_free(b);
    le_tensor_free(a);
    
    return EXIT_SUCCESS;
}
