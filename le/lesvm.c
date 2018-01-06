#include "lesvm.h"
#include <stdlib.h>

LeSVM *
le_svm_new_train(LeMatrix *x_train, LeMatrix *y_train)
{
    LeSVM *self = NULL;
    
    return self;
}

LeMatrix *
le_svm_prefict(LeSVM *svm, LeMatrix *x)
{
    LeMatrix *y = le_matrix_new_zeros(1, le_matrix_get_width(x));
    
    return y;
}

void
le_svm_free(LeSVM *svm)
{
    free(svm);
}
