#include <stdlib.h>
#include "lelogistic.h"

struct LeLogisticClassifier
{
    LeMatrix *w;
    float     b;
    unsigned  polynomia_degree;
};

LeLogisticClassifier *
le_logistic_classifier_new_train(LeMatrix *x_train, LeMatrix *y_train, unsigned polynomia_degree)
{
    LeLogisticClassifier *self = NULL;
    unsigned examples_count = le_matrix_get_width(x_train);
    unsigned iterations_count = 200;
    float learning_rate = 0.1;
    unsigned i;
    
    if (le_matrix_get_width(y_train) != examples_count)
    {
        return NULL;
    }
    
    LeMatrix *x = x_train;
    LeMatrix *x_prev = x_train;
    
    for (i = 0; i < polynomia_degree; i++)
    {
        x = le_matrix_new_polynomia(x_prev);
        if (x_prev != x_train)
        {
            le_matrix_free(x_prev);
        }
        x_prev = x;
    }
    
    unsigned features_count = le_matrix_get_height(x);
    LeMatrix *xt = le_matrix_new_transpose(x);
    
    if (x != x_train)
    {
        le_matrix_free(x);
    }
    
    self = malloc(sizeof(struct LeLogisticClassifier));
    self->w = le_matrix_new_zeros(features_count, 1);
    self->b = 0;
    self->polynomia_degree = polynomia_degree;
    
    for (i = 0; i < iterations_count; i++)
    {
        LeMatrix *h = le_logistic_classifier_prefict(self, x_train);
        le_matrix_subtract(h, y_train);
        le_matrix_multiply_by_scalar(h, 1.0 / examples_count);
        LeMatrix *dwt = le_matrix_new_product(h, xt);
        LeMatrix *dw = le_matrix_new_transpose(dwt);
        le_matrix_multiply_by_scalar(dw, learning_rate);
        float db = le_matrix_sum(h);
        
        le_matrix_free(dwt);
        le_matrix_free(h);
        le_matrix_subtract(self->w, dw);
        le_matrix_free(dw);
        self->b -= learning_rate * db;
    }
    
    le_matrix_free(xt);
    
    return self;
}

LeMatrix *
le_logistic_classifier_prefict(LeLogisticClassifier *self, LeMatrix *x)
{
    unsigned i;
    LeMatrix *wt = le_matrix_new_transpose(self->w);
    LeMatrix *x_poly = x;
    LeMatrix *x_prev = x;
    for (i = 0; i < self->polynomia_degree; i++)
    {
        x_poly = le_matrix_new_polynomia(x_prev);
        if (x_prev != x)
        {
            le_matrix_free(x_prev);
        }
        x_prev = x_poly;
    }
    LeMatrix *a = le_matrix_new_product(wt, x_poly);
    le_matrix_free(wt);
    if (x_poly != x)
    {
        le_matrix_free(x_poly);
    }
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
