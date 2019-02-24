/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lesvm.h"
#include <stdlib.h>
#include "lemodel.h"

struct LeSVM
{
    LeModel   parent;
    
    /* Training data */
    LeMatrix *x;
    LeMatrix *y;
    
    LeKernel  kernel;
    float     bias;
    /* Weights for linear classifier */
    LeMatrix *weights;
    LeMatrix *alphas;
};

typedef struct LeSVMClass
{
    LeModelClass parent;
} LeSVMClass;

LeSVMClass le_svm_class;

LeMatrix * le_svm_predict(LeSVM *self, LeMatrix *x);

void
le_svm_class_ensure_init(void)
{
    static int le_svm_class_initialized = 0;
    
    if (!le_svm_class_initialized)
    {
        le_svm_class.parent.predict =
        (LeMatrix *(*)(LeModel *, LeMatrix *))le_svm_predict;
        le_svm_class_initialized = 1;
    }
}

void
le_svm_construct(LeSVM *self)
{
    le_model_construct((LeModel *)self);
    le_svm_class_ensure_init();
    ((LeObject *)self)->klass = (LeClass *)&le_svm_class;
    self->x = NULL;
    self->y = NULL;
    self->bias = 0.0f;
    self->alphas = NULL;
    self->weights = NULL;
    self->kernel = LE_KERNEL_LINEAR;
}

LeSVM *
le_svm_new(void)
{
    LeSVM *self = malloc(sizeof(struct LeSVM));
    le_svm_construct(self);
    return self;
}

void
le_svm_train(LeSVM *self, LeMatrix *x_train, LeMatrix *y_train, LeKernel kernel)
{
    unsigned passes = 0;
    /// @todo: Expose this parameter
    unsigned max_passes = 127;

    /// @todo: Add checks
    self->x = le_matrix_new_copy(x_train);
    self->y = le_matrix_new_copy(y_train);
    self->kernel = kernel;
    /// @todo: Sequential Minimal Optimization here
    self->alphas = le_matrix_new_zeros(1, le_matrix_get_width(x_train));
    self->bias = 0;
    
    while (passes < max_passes)
    {
        unsigned num_changed_alphas = 0;
        
        if (num_changed_alphas == 0)
            passes += 1;
        else
            passes = 0;
    }
}

float
le_svm_kernel(LeMatrix *a, LeMatrix *b, LeKernel kernel)
{
    switch (kernel) {
    case LE_KERNEL_RBF:
        return 0.0f;
    case LE_KERNEL_LINEAR:
    default:
        return le_dot_product(a, b);
    }
}

static float
le_svm_function(LeSVM *self, LeMatrix *x)
{
    unsigned i;
    float result = 0;
    unsigned training_examples_count = le_matrix_get_width(self->x);

    for (i = 0; i < training_examples_count; i++)
    {
        result += le_matrix_at(self->alphas, 0, i) * le_matrix_at(self->y, 0, i) * le_svm_kernel(self->x, x, self->kernel);
    }
    result += self->bias;
    return result;
}

LeMatrix *
le_svm_predict(LeSVM *self, LeMatrix *x)
{
    unsigned i;
    unsigned examples_count;
    
    if (self == NULL)
        return NULL;
    
    /* In case we use linear kernel and have weights, apply linear classification */
    if (self->weights != NULL)
    {
        LeMatrix *wt = le_matrix_new_transpose(self->weights);
        LeMatrix *y_predicted = le_matrix_new_product(wt, x);
        le_matrix_free(wt);
        le_matrix_add_scalar(y_predicted, self->bias);
    }
    
    if (self->alphas == NULL)
        return NULL;
    
    examples_count = le_matrix_get_width(x);
    
    LeMatrix *y_pred = le_matrix_new_uninitialized(1, examples_count);
    for (i = 0; i < examples_count; i++)
    {
        LeMatrix *example = le_matrix_get_column(x, i);
        le_matrix_set_element(y_pred, 0, i, le_svm_function(self, example));
        le_matrix_free(example);
    }

    le_matrix_apply_greater_than(y_pred, 0);
    return y_pred;
}

void
le_svm_free(LeSVM *self)
{
    free(self);
}
