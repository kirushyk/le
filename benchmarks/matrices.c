/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <le/tensors/lematrix.h>

#define MIN_DIMENSION 16
#define MAX_DIMENSION 48

int
main()
{
    unsigned width;
    unsigned height;
    unsigned second_width;
    
    LeTensor *a;
    LeTensor *b;
    LeTensor *c;
    
    for (height = MIN_DIMENSION; height <= MAX_DIMENSION; height++)
    {
        for (width = MIN_DIMENSION; width <= MAX_DIMENSION; width++)
        {
            a = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, height, width);
            
            for (second_width = MIN_DIMENSION; second_width <= MAX_DIMENSION; second_width++)
            {
                b = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, width, second_width);
                c = le_matrix_new_product(a, b);
                le_tensor_unref(c);
                le_tensor_unref(b);
            }
            
            le_tensor_unref(a);
        }
    }
    return EXIT_SUCCESS;
}
