/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdio.h>
#include <math.h>
#include <le/le.h>

int
main(int argc, const char *argv[])
{
    const float x_data[] =
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        4.0f, 3.0f, 2.0f, 1.0f
    };
    LeMatrix *x = le_matrix_new_from_data(2, 4, x_data);
    
    const float y_data[] =
    {
        -1.0f, -1.0f, 1.0f, 1.0f
    };
    LeMatrix *y = le_matrix_new_from_data(1, 4, y_data);
    
    LeSVM *svm = le_svm_new();
    le_svm_train(svm, x, y, LE_KERNEL_LINEAR);
    
    LeMatrix *h = le_model_predict((LeModel *)svm, x);
    printf("Preficted value =\n");
    le_matrix_print(h, stdout);
    
    le_matrix_free(h);
    le_matrix_free(y);
    le_matrix_free(x);
    
    return 0;
}

