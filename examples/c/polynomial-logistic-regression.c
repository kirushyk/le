/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdio.h>
#include <math.h>
#include <le/le.h>

int
main(int argc, const char *argv[])
{
    LeTensor *x = le_tensor_new(LE_TYPE_FLOAT32, 2, 2, 4,
        1.0, 2.0, 3.0, 4.0,
        4.0, 3.0, 2.0, 1.0
    );
    
    LeTensor *y = le_tensor_new(LE_TYPE_FLOAT32, 2, 1, 4,
        0.0, 0.0, 1.0, 1.0
    );
    
    printf("Train set:\n");
    printf("x =\n");
    le_tensor_print(x, stdout);
    printf("y =\n");
    le_tensor_print(y, stdout);
    
    LeLogisticClassifier *lc = le_logistic_classifier_new();
    LeLogisticClassifierTrainingOptions options;
    options.max_iterations = 100;
    options.learning_rate = 1.0f;
    options.polynomia_degree = 1;
    options.regularization = LE_REGULARIZATION_NONE;
    options.lambda = 0.0f;
    le_logistic_classifier_train(lc, x, y, options);
    
    LeTensor *h = le_model_predict(LE_MODEL(lc), x);
    printf("Predicted value =\n");
    le_tensor_print(h, stdout);
    
    le_tensor_unref(h);
    le_tensor_unref(y);
    le_tensor_unref(x);
    
    return 0;
}
