/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdio.h>
#include <math.h>
#include <le/le.h>

int
main(int argc, const char *argv[])
{
    LeTensor *x = le_tensor_new(LE_TYPE_F32, 2, 2, 4,
        1.0, 2.0, 3.0, 4.0,
        4.0, 3.0, 2.0, 1.0
    );
    
    LeTensor *y = le_tensor_new(LE_TYPE_F32, 2, 1, 4,
        -1.0, -1.0, 1.0, 1.0
    );
    
    printf("Train set:\n");
    printf("x =\n");
    le_tensor_print(x, stdout);
    printf("y =\n");
    le_tensor_print(y, stdout);

    LeSVM *svm = le_svm_new();
    LeSVMTrainingOptions options;
    options.kernel = LE_KERNEL_LINEAR;
    options.c = 1.0f;
    le_svm_train(svm, x, y, options);
    
    LeTensor *h = le_model_predict(LE_MODEL(svm), x);
    printf("Predicted value =\n");
    le_tensor_print(h, stdout);
    
    le_tensor_unref(h);
    le_tensor_unref(y);
    le_tensor_unref(x);
    
    return 0;
}

