#include <stdlib.h>
#include "lelogistic.h"

struct LeLogisticClassifier
{
    LeMatrix *w;
    double    b;
};

LeLogisticClassifier *
le_logistic_classifier_new_train(LeMatrix *x_train, LeMatrix *y_train)
{
    LeLogisticClassifier *self = NULL;
    unsigned features_count = le_matrix_get_height(x_train);
    unsigned examples_count = le_matrix_get_width(x_train);
    if (le_matrix_get_width(y_train) == examples_count)
    {
        self = malloc(sizeof(struct LeLogisticClassifier));
        self->w = le_matrix_new_zeros(features_count, 1);
        self->b = 0;
    }
    return self;
}

LeMatrix *
le_logistic_classifier_prefict(LeLogisticClassifier *self, LeMatrix *x)
{
    LeMatrix *wt = le_matrix_new_transpose(self->w);
    LeMatrix *a = le_matrix_new_product(wt, x);
    le_matrix_free(wt);
    le_matrix_add_scalar(a, self->b);
    le_matrix_apply_sigmoid(a);
    return a;
}

void
le_logistic_classifier_free(LeLogisticClassifier *self)
{
    le_matrix_free(self->w);
    free(self);
}
