/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "le1layernn.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "lemodel.h"
#include <le/tensors/letensor-imp.h>
#include <le/tensors/lematrix.h>
#include "leloss.h"

struct Le1LayerNN
{
    LeModel   parent;
    LeTensor *weights;
    LeTensor *bias;

};

typedef struct Le1LayerNNClass
{
    LeModelClass parent;
} Le1LayerNNClass;

LeTensor *
le_1_layer_nn_predict(Le1LayerNN *self, const LeTensor *x);

GList *
le_1_layer_nn_get_gradients(Le1LayerNN *self, const LeTensor *x, const LeTensor *y);

Le1LayerNNClass *
le_1_layer_nn_class_ensure_init(void)
{
    static int le_1_layer_nn_class_initialized = 0;
    static Le1LayerNNClass klass;

    if (!le_1_layer_nn_class_initialized)
    {
        klass.parent.predict =
            (LeTensor *(*)(LeModel *, const LeTensor *))le_1_layer_nn_predict;
        klass.parent.get_gradients =
            (GList *(*)(LeModel *, const LeTensor *, const LeTensor *))le_1_layer_nn_get_gradients;
        le_1_layer_nn_class_initialized = 1;
    }

    return &klass;
}

void
le_1_layer_nn_construct(Le1LayerNN *self)
{
    le_model_construct((LeModel *)self);
    G_OBJECT_GET_CLASS(self) = G_OBJECT_CLASS(le_1_layer_nn_class_ensure_init());
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
le_1_layer_nn_predict(Le1LayerNN *self, const LeTensor *x)
{
    LeTensor *a = le_matrix_new_product(self->weights, x);
    le_matrix_add(a, self->bias);
    le_tensor_apply_sigmoid(a);
    // le_matrix_apply_softmax(a);
    return a;
}

GList *
le_1_layer_nn_get_gradients(Le1LayerNN *self, const LeTensor *x, const LeTensor *y)
{
    unsigned examples_count = le_matrix_get_width(x);

    LeTensor *h = le_1_layer_nn_predict(self, x);
    le_tensor_sub(h, y);
    le_tensor_mul(h, 1.0f / examples_count);
    LeTensor *dw = le_matrix_new_product_full(h, false, x, true);
    LeTensor *db = le_matrix_new_sum(h, 1);
    le_tensor_free(h);

    GList *gradients = NULL;
    
    gradients = g_list_append(gradients, dw);
    gradients = g_list_append(gradients, db);

    return gradients;
}

void
le_1_layer_nn_init(Le1LayerNN *self, unsigned features_count, unsigned classes_count)
{
    assert(self);
    assert(self->bias == NULL);
    assert(self->weights == NULL);
    assert(features_count >= 1);
    assert(classes_count >= 1);
    
    self->weights = le_matrix_new_rand_f32(LE_DISTRIBUTION_NORMAL, classes_count, features_count);
    le_tensor_mul_f32(self->weights, sqrtf(1.0f / features_count));
    self->bias = le_matrix_new_zeros(LE_TYPE_FLOAT32, classes_count, 1);
    
    le_model_append_parameter(LE_MODEL(self), self->weights);
    le_model_append_parameter(LE_MODEL(self), self->bias);
}

void
le_1_layer_nn_train(Le1LayerNN *self, LeTensor *x_train, LeTensor *y_train, Le1LayerNNTrainingOptions options)
{
    assert(self);
    assert(self->bias);
    assert(self->weights);

    unsigned examples_count = le_matrix_get_width(x_train);
    unsigned classes_count = le_matrix_get_height(y_train);
    unsigned iterations_count = options.max_iterations;
    unsigned i;
    
    assert(le_matrix_get_width(y_train) == examples_count);
    
    unsigned features_count = le_matrix_get_height(x_train);
    LeTensor *xt = le_matrix_new_transpose(x_train);
    
    assert(le_matrix_get_width(self->weights) == features_count);
    assert(le_matrix_get_height(self->weights) == classes_count);
    assert(le_matrix_get_height(self->bias) == classes_count);
    
    for (i = 0; i < iterations_count; i++)
    {
        printf("Iteration %u. ", i);

        LeTensor *h = le_1_layer_nn_predict(self, x_train);

        float train_set_error = le_logistic_loss(h, y_train);

        le_tensor_sub(h, y_train);
        le_tensor_mul(h, 1.0f / examples_count);
        LeTensor *dw = le_matrix_new_product(h, xt);
        le_tensor_mul(dw, options.learning_rate);
        LeTensor *db = le_matrix_new_sum(h, 1);
        
        le_tensor_free(h);
        le_tensor_sub(self->weights, dw);
        le_tensor_free(dw);
        le_tensor_sub_scaled_f32(self->bias, options.learning_rate, db);
        le_tensor_free(db);
        
        printf("Train Set Error: %f\n", train_set_error);
    }
    
    le_tensor_free(xt);
}

void
le_1_layer_nn_free(Le1LayerNN *self)
{
    le_tensor_free(self->weights);
    le_tensor_free(self->bias);
    g_free (self);
}
