/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lesvm.h"
#include <stdlib.h>

struct LeSVM
{
    LeMatrix *w;
    float     b;
    float     c;
    LeKernel  kernel;
};

LeSVM *
le_svm_new(void)
{
    LeSVM *self = NULL;
    
    return self;
}

void
le_svm_train(LeSVM *self, LeMatrix *x_train, LeMatrix *y_train, LeKernel kernel)
{
    /// @todo: Sequential Minimal Optimization here
}

LeMatrix *
le_svm_predict(LeSVM *self, LeMatrix *x)
{
    LeMatrix *wt = le_matrix_new_transpose(self->w);
    LeMatrix *a = le_matrix_new_product(wt, x);
    le_matrix_free(wt);
    le_matrix_add_scalar(a, self->b);
    le_matrix_apply_greater_than(a, 0);
    return a;
}

void
le_svm_free(LeSVM *self)
{
    free(self);
}
