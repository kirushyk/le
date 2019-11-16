/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lelogistic.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "lemodel.h"
#include "letensor-imp.h"
#include "lematrix.h"
#include "lepolynomia.h"

struct LeLogisticClassifier
{
    LeModel   parent;
    LeTensor *weights;
    float     bias;
    unsigned  polynomia_degree;
};

typedef struct LeLogisticClassifierClass
{
    LeModelClass parent;
} LeLogisticClassifierClass;

LeLogisticClassifierClass le_logistic_classifier_class;

LeTensor * le_logistic_classifier_predict(LeLogisticClassifier *self, LeTensor *x);

void
le_logistic_classifier_class_ensure_init(void)
{
    static int le_logistic_classifier_class_initialized = 0;

    if (!le_logistic_classifier_class_initialized)
    {
        le_logistic_classifier_class.parent.predict =
            (LeTensor *(*)(LeModel *, LeTensor *))le_logistic_classifier_predict;
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

LeTensor *
le_logistic_classifier_predict(LeLogisticClassifier *self, LeTensor *x)
{
    unsigned i;
    LeTensor *wt = le_matrix_new_transpose(self->weights);
    LeTensor *x_poly = x;
    LeTensor *x_prev = x;
    for (i = 0; i < self->polynomia_degree; i++)
    {
        x_poly = le_matrix_new_polynomia(x_prev);
        if (x_prev != x)
        {
            le_tensor_free(x_prev);
        }
        x_prev = x_poly;
    }
    LeTensor *a = le_matrix_new_product(wt, x_poly);
    le_tensor_free(wt);
    if (x_poly != x)
    {
        le_tensor_free(x_poly);
    }
    le_tensor_add_scalar(a, self->bias);
    le_tensor_apply_sigmoid(a);
    return a;
}

float
logistic_error(LeTensor *h, LeTensor *y)
{
    assert(h->shape->num_dimensions == 2);
    assert(y->shape->num_dimensions == 2);
    assert(h->shape->sizes[1] == y->shape->sizes[1]);
    
    float result = 0.0f;
    unsigned i;
    
    unsigned elements_count = le_shape_get_elements_count(h->shape);
    for (i = 0; i < elements_count; i++)
    {
        float yi = le_tensor_f32_at(y, i);
        float hi = le_tensor_f32_at(h, i);
        result -= yi * log(hi) + (1.0f - yi) * log(1.0f - hi);
    }
    
    return result / elements_count;
}

void
le_logistic_classifier_train(LeLogisticClassifier *self, LeTensor *x_train, LeTensor *y_train, LeLogisticClassifierTrainingOptions options)
{
    unsigned examples_count = le_matrix_get_width(x_train);
    unsigned iterations_count = options.max_iterations;
    unsigned i;
    
    assert(le_matrix_get_width(y_train) == examples_count);
    
    LeTensor *x = x_train;
    LeTensor *x_prev = x_train;
    
    for (i = 0; i < options.polynomia_degree; i++)
    {
        x = le_matrix_new_polynomia(x_prev);
        if (x_prev != x_train)
        {
            le_tensor_free(x_prev);
        }
        x_prev = x;
    }
    
    unsigned features_count = le_matrix_get_height(x);
    LeTensor *xt = le_matrix_new_transpose(x);
    
    if (x != x_train)
    {
        le_tensor_free(x);
    }
    
    self->weights = le_matrix_new_zeros(features_count, 1);
    self->bias = 0;
    self->polynomia_degree = options.polynomia_degree;
    
    for (i = 0; i < iterations_count; i++)
    {
        printf("Iteration %u. ", i);
        
        LeTensor *h = le_logistic_classifier_predict(self, x_train);
        
        float train_set_error = logistic_error(h, y_train);
        
        le_tensor_subtract(h, y_train);
        le_tensor_multiply_by_scalar(h, 1.0 / examples_count);
        LeTensor *dwt = le_matrix_new_product(h, xt);
        LeTensor *dw = le_matrix_new_transpose(dwt);
        le_tensor_multiply_by_scalar(dw, options.alpha);
        float db = le_tensor_sum(h);
        
        le_tensor_free(dwt);
        le_tensor_free(h);
        le_tensor_subtract(self->weights, dw);
        le_tensor_free(dw);
        self->bias -= options.alpha * db;
        
        printf("Train Set Error: %f\n", train_set_error);
    }
    
    le_tensor_free(xt);
}

void
le_logistic_classifier_free(LeLogisticClassifier *self)
{
    le_tensor_free(self->weights);
    free(self);
}
