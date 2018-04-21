/** Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
  * Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdio.h>
#include <le/le.h>

int
main(int argc, const char *argv[])
{
    LeMatrix *a = le_matrix_new_rand(2, 4);
    printf("a =\n");
    le_matrix_print(a, stdout);
    
    LeMatrix *b = le_matrix_new_polynomia(a);
    printf("poly(a) =\n");
    le_matrix_print(b, stdout);
    
    LeMatrix *c = le_matrix_new_polynomia(b);
    printf("poly(poly(a)) =\n");
    le_matrix_print(c, stdout);
    
    le_matrix_free(c);
    le_matrix_free(b);
    le_matrix_free(a);
    
    return 0;
}
