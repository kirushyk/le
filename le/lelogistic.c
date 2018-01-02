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
    unsigned iterations_count = 200;
    double learning_rate = 0.1;
    unsigned i;
    
    if (le_matrix_get_width(y_train) == examples_count)
    {
        LeMatrix *xt = le_matrix_new_transpose(x_train);
        
        self = malloc(sizeof(struct LeLogisticClassifier));
        self->w = le_matrix_new_zeros(features_count, 1);
        self->b = 0;
        
        for (i = 0; i < iterations_count; i++)
        {
            LeMatrix *h = le_logistic_classifier_prefict(self, x_train);
            le_matrix_subtract(h, y_train);
            le_matrix_multiply_by_scalar(h, 1.0 / examples_count);
            LeMatrix *dwt = le_matrix_new_product(h, xt);
            LeMatrix *dw = le_matrix_new_transpose(dwt);
            double db = le_matrix_sum(h);
            
            le_matrix_free(dwt);
            le_matrix_free(h);
            le_matrix_subtract(self->w, dw);
            le_matrix_free(dw);
            self->b -= learning_rate * db;
        }
        
        le_matrix_free(xt);
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
