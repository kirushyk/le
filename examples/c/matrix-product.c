/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdio.h>
#include <le/le.h>

int
main(int argc, const char *argv[])
{
    LeTensor *a = le_matrix_new_rand(4, 2);
    printf("a =\n");
    le_tensor_print(a, stdout);
    
    LeTensor *b = le_matrix_new_rand(2, 4);
    printf("b =\n");
    le_tensor_print(b, stdout);
    
    LeTensor *c = le_matrix_new_product(a, b);
    printf("c = a * b =\n");
    le_tensor_print(c, stdout);
    
    LeTensor *d = le_matrix_new_transpose(c);
    printf("d = c' =\n");
    le_tensor_print(d, stdout);
    
    le_tensor_free(d);
    le_tensor_free(c);
    le_tensor_free(b);
    le_tensor_free(a);
    
    return 0;
}
