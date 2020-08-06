/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lelogistic.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "lemodel.h"
#include <le/tensors/letensor-imp.h>
#include <le/tensors/lematrix.h>
#include "lepolynomia.h"
#include "leloss.h"

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

LeTensor *
le_logistic_classifier_predict(LeLogisticClassifier *self, const LeTensor *x);

LeLogisticClassifierClass *
le_logistic_classifier_class_ensure_init(void)
{
    static bool initialized = false;
    static LeLogisticClassifierClass klass;

    if (!initialized)
    {
        klass.parent.predict =
            (LeTensor *(*)(LeModel *, const LeTensor *))le_logistic_classifier_predict;
        initialized = 1;
    }

    return &klass;
}

void
le_logistic_classifier_construct(LeLogisticClassifier *self)
{
    le_model_construct(LE_MODEL(self));
    LE_OBJECT_GET_CLASS(self) = LE_CLASS(le_logistic_classifier_class_ensure_init());
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
le_logistic_classifier_predict(LeLogisticClassifier *self, const LeTensor *x)
{
    unsigned i;
    LeTensor *wt = le_matrix_new_transpose(self->weights);
    LeTensor *x_prev = NULL;
    LeTensor *x_poly = NULL;
    for (i = 0; i < self->polynomia_degree; i++)
    {
        /// @note: Refrain from init x_prev = x to prevent const
        x_poly = le_matrix_new_polynomia(x_prev ? x_prev : x);
        le_tensor_free(x_prev);
        x_prev = x_poly;
    }
    LeTensor *a = le_matrix_new_product(wt, x_poly ? x_poly : x);
    le_tensor_free(wt);
    if (x_poly != x)
    {
        le_tensor_free(x_poly);
    }
    le_tensor_add(a, self->bias);
    le_tensor_apply_sigmoid(a);
    return a;
}

void
le_logistic_classifier_train(LeLogisticClassifier *self, const LeTensor *x_train, const LeTensor *y_train, LeLogisticClassifierTrainingOptions options)
{
    unsigned examples_count = le_matrix_get_width(x_train);
    unsigned iterations_count = options.max_iterations;
    unsigned i;
    
    assert(le_matrix_get_width(y_train) == examples_count);
    
    const LeTensor *x = x_train;
    const LeTensor *x_prev = x_train;
    
    for (i = 0; i < options.polynomia_degree; i++)
    {
        x = le_matrix_new_polynomia(x_prev);
        if (x_prev != x_train)
        {
            le_tensor_free(LE_TENSOR(x_prev));
        }
        x_prev = x;
    }
    
    unsigned features_count = le_matrix_get_height(x);
    
    /*
    if (x != x_train)
    {
        le_tensor_free(x);
    }
    */
    
    self->weights = le_matrix_new_zeros(LE_TYPE_FLOAT32, features_count, 1);
    self->bias = 0;
    self->polynomia_degree = options.polynomia_degree;
    
    for (i = 0; i < iterations_count; i++)
    {
        printf("Iteration %u. ", i);
        
        LeTensor *h = le_logistic_classifier_predict(self, x_train);
        
        float train_set_error = le_logistic_loss(h, y_train);
        
        le_tensor_sub(h, y_train);
        le_tensor_mul(h, 1.0f / examples_count);
        LeTensor *dwt = le_matrix_new_product_full(h, false, x, true);
        LeTensor *dw = le_matrix_new_transpose(dwt);
        le_tensor_mul(dw, options.learning_rate);
        float db = le_tensor_sum_f32(h);
        
        le_tensor_free(dwt);
        le_tensor_free(h);
        le_tensor_sub(self->weights, dw);
        le_tensor_free(dw);
        self->bias -= options.learning_rate * db;
        
        printf("Train Set Error: %f\n", train_set_error);
    }
}

void
le_logistic_classifier_free(LeLogisticClassifier *self)
{
    le_tensor_free(self->weights);
    free(self);
}
