/** Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
  * Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lesvm.h"
#include <stdlib.h>

LeSVM *
le_svm_new_train(LeMatrix *x_train, LeMatrix *y_train)
{
    LeSVM *self = NULL;
    
    return self;
}

LeMatrix *
le_svm_predict(LeSVM *svm, LeMatrix *x)
{
    LeMatrix *y = le_matrix_new_zeros(1, le_matrix_get_width(x));
    
    return y;
}

void
le_svm_free(LeSVM *svm)
{
    free(svm);
}
