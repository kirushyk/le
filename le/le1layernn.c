/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "le1layernn.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "lemodel.h"
#include "letensor-imp.h"
#include "lematrix.h"
#include "lepolynomia.h"

struct Le1LayerNN
{
    LeModel   parent;
    LeTensor *weights;
    float     bias;
};

typedef struct Le1LayerNNClass
{
    LeModelClass parent;
} Le1LayerNNClass;

Le1LayerNNClass le_1_layer_nn_class;

LeTensor * le_1_layer_nn_predict(Le1LayerNN *self, LeTensor *x);

void
le_1_layer_nn_class_ensure_init(void)
{
    static int le_1_layer_nn_class_initialized = 0;

    if (!le_1_layer_nn_class_initialized)
    {
        le_1_layer_nn_class.parent.predict =
            (LeTensor *(*)(LeModel *, LeTensor *))le_1_layer_nn_predict;
        le_1_layer_nn_class_initialized = 1;
    }
}

void
le_1_layer_nn_construct(Le1LayerNN *self)
{
    le_model_construct((LeModel *)self);
    le_1_layer_nn_class_ensure_init();
    ((LeObject *)self)->klass = (LeClass *)&le_1_layer_nn_class;
    self->weights = NULL;
    self->bias = 0;
}

Le1LayerNN *
le_1_layer_nn_new(void)
{
    Le1LayerNN *self = malloc(sizeof(struct Le1LayerNN));
    le_1_layer_nn_construct(self);
    return self;
}

LeTensor *
le_1_layer_nn_predict(Le1LayerNN *self, LeTensor *x)
{
    LeTensor *a = le_matrix_new_product(self->weights, x);
    le_tensor_add_scalar(a, self->bias);
    le_tensor_apply_sigmoid(a);
    return a;
}

float
logistic_error(LeTensor *h, LeTensor *y);

void
le_1_layer_nn_train(Le1LayerNN *self, LeTensor *x_train, LeTensor *y_train, Le1LayerNNTrainingOptions options)
{
    unsigned examples_count = le_matrix_get_width(x_train);
    unsigned classes_count = le_matrix_get_height(y_train);
    unsigned iterations_count = options.max_iterations;
    unsigned i;
    
    assert(le_matrix_get_width(y_train) == examples_count);
    
    unsigned features_count = le_matrix_get_height(x_train);
    LeTensor *xt = le_matrix_new_transpose(x_train);
    
    self->weights = le_matrix_new_zeros(classes_count, features_count);
    self->bias = 0;
    
    for (i = 0; i < iterations_count; i++)
    {
        printf("Iteration %u. ", i);

        LeTensor *h = le_1_layer_nn_predict(self, x_train);

        float train_set_error = logistic_error(h, y_train);

        le_tensor_subtract(h, y_train);
        le_tensor_multiply_by_scalar(h, 1.0 / examples_count);
        LeTensor *dw = le_matrix_new_product(h, xt);
        le_tensor_multiply_by_scalar(dw, options.alpha);
        float db = le_tensor_sum(h);
        
        le_tensor_free(h);
        le_tensor_subtract(self->weights, dw);
        le_tensor_free(dw);
        self->bias -= options.alpha * db;
        
        printf("Train Set Error: %f\n", train_set_error);
    }
    
    le_tensor_free(xt);
}

void
le_1_layer_nn_free(Le1LayerNN *self)
{
    le_tensor_free(self->weights);
    free(self);
}
