/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lelogistic.h"
#include <stdlib.h>
#include "lemodel.h"

struct LeLogisticClassifier
{
    LeModel   parent;
    LeMatrix *weights;
    float     bias;
    unsigned  polynomia_degree;
};

typedef struct LeLogisticClassifierClass
{
    LeModelClass parent;
} LeLogisticClassifierClass;

LeLogisticClassifierClass le_logistic_classifier_class;

LeMatrix * le_logistic_classifier_predict(LeLogisticClassifier *self, LeMatrix *x);

void
le_logistic_classifier_class_ensure_init(void)
{
    static int le_logistic_classifier_class_initialized = 0;

    if (!le_logistic_classifier_class_initialized)
    {
        le_logistic_classifier_class.parent.predict =
            (LeMatrix *(*)(LeModel *, LeMatrix *))le_logistic_classifier_predict;
        le_logistic_classifier_class_initialized = 1;
    }
}

void
le_logistic_classifier_construct(LeLogisticClassifier *self)
{
    le_model_construct((LeModel *)self);
    le_logistic_classifier_class_ensure_init();
    ((LeObject *)self)->klass = (LeClass *)&le_logistic_classifier_class;
    self->weights = NULL;
    self->bias = 0;
    self->polynomia_degree = 0;
}

LeLogisticClassifier *
le_logistic_classifier_new(void)
{
    LeLogisticClassifier *self = malloc(sizeof(struct LeLogisticClassifier));
    le_logistic_classifier_construct(self);
    return self;
}

LeMatrix *
le_logistic_classifier_predict(LeLogisticClassifier *self, LeMatrix *x)
{
    unsigned i;
    LeMatrix *wt = le_matrix_new_transpose(self->weights);
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
    le_matrix_add_scalar(a, self->bias);
    le_matrix_apply_sigmoid(a);
    return a;
}

void
le_logistic_classifier_train(LeLogisticClassifier *self, LeMatrix *x_train, LeMatrix *y_train, LeLogisticClassifierTrainingOptions options)
{
    unsigned examples_count = le_matrix_get_width(x_train);
    unsigned iterations_count = 200;
    float learning_rate = 0.1;
    unsigned i;
    
    if (le_matrix_get_width(y_train) != examples_count)
    {
        /** @todo: Handle this case */
        return;
    }
    
    LeMatrix *x = x_train;
    LeMatrix *x_prev = x_train;
    
    for (i = 0; i < options.polynomia_degree; i++)
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
    
    self->weights = le_matrix_new_zeros(features_count, 1);
    self->bias = 0;
    self->polynomia_degree = options.polynomia_degree;
    
    for (i = 0; i < iterations_count; i++)
    {
        LeMatrix *h = le_logistic_classifier_predict(self, x_train);
        le_matrix_subtract(h, y_train);
        le_matrix_multiply_by_scalar(h, 1.0 / examples_count);
        LeMatrix *dwt = le_matrix_new_product(h, xt);
        LeMatrix *dw = le_matrix_new_transpose(dwt);
        le_matrix_multiply_by_scalar(dw, learning_rate);
        float db = le_matrix_sum(h);
        
        le_matrix_free(dwt);
        le_matrix_free(h);
        le_matrix_subtract(self->weights, dw);
        le_matrix_free(dw);
        self->bias -= learning_rate * db;
    }
    
    le_matrix_free(xt);
}

void
le_logistic_classifier_free(LeLogisticClassifier *self)
{
    le_matrix_free(self->weights);
    free(self);
}
