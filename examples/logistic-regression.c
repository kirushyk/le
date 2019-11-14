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
    LeTensor *x = le_matrix_new_from_data(2, 4, x_data);
    
    const float y_data[] =
    {
        0.0f, 0.0f, 1.0f, 1.0f
    };
    LeTensor *y = le_matrix_new_from_data(1, 4, y_data);
    
    printf("Train set:\n");
    printf("x =\n");
    le_matrix_print(x, stdout);
    printf("y =\n");
    le_matrix_print(y, stdout);
    
    LeLogisticClassifier *lc = le_logistic_classifier_new();
    LeLogisticClassifierTrainingOptions options;
    options.max_iterations = 100;
    options.alpha = 1.0f;
    options.polynomia_degree = 1;
    options.regularization = LE_REGULARIZATION_NONE;
    options.lambda = 0.0f;
    le_logistic_classifier_train(lc, x, y, options);
    
    LeTensor *h = le_model_predict((LeModel *)lc, x);
    printf("Predicted value =\n");
    le_matrix_print(h, stdout);
    
    le_tensor_free(h);
    le_tensor_free(y);
    le_tensor_free(x);
    
    return 0;
}
