/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lesvm.h"
#include <assert.h>
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
    
    unsigned features_count = le_matrix_get_height(x_train);
    unsigned examples_count = le_matrix_get_width(x_train);
    /// @todo: Add more clever input data checks
    assert(examples_count == le_matrix_get_width(y_train));


    /// @todo: Add checks
    self->x = NULL;
    self->y = NULL;
    self->kernel = kernel;
    /// @todo: Add cleanup here
    self->alphas = NULL;
    /// @note: Maybe use stack variable instead
    LeMatrix *alphas = le_matrix_new_rand(1, examples_count);
    self->bias = 0;
    /// @todo: Add cleanup here
    self->weights = NULL;
    
    /// @note: Sequential Minimal Optimization (SMO) algorithm
    while (passes < max_passes)
    {
        unsigned num_changed_alphas = 0;
        
        if (num_changed_alphas == 0)
            passes++;
        else
            passes = 0;
    }
    
    if (kernel == LE_KERNEL_LINEAR)
    {
        /* For linear kernel, we calculate weights */
        self->weights = le_matrix_new_uninitialized(features_count, 1);
        for (int j = 0; j < features_count; j++)
        {
            float s = 0.0f;
            for (int i = 0; i < examples_count; i++)
            {
                s += le_matrix_at(alphas, i, 0) * le_matrix_at(y_train, i, 0) * le_matrix_at(x_train, j, i);
            }
            le_matrix_set_element(self->weights, j, 0, s);
        }
    }
    else
    {
        /* For other kernels, we only retain alphas and training data for support vectors */
        unsigned support_vectors_count = 0;
        const float alpha_tolerance = 1e-4;
        for (int i = 0; i < examples_count; i++)
        {
            if (le_matrix_at(alphas, 0, i) >= alpha_tolerance)
                support_vectors_count++;
        }
        
        self->alphas = le_matrix_new_uninitialized(1, support_vectors_count);
        self->x = le_matrix_new_uninitialized(features_count, support_vectors_count);
        self->y = le_matrix_new_uninitialized(1, support_vectors_count);

        int j = 0; /// Iterator for new matrices
        for (int i = 0; i < examples_count; i++)
        {
            if (le_matrix_at(alphas, 0, i) >= alpha_tolerance)
            {
                le_matrix_set_element(self->alphas, 0, j, le_matrix_at(alphas, 0, i));
                le_matrix_set_element(self->y, 0, j, le_matrix_at(y_train, 0, i));
                for (int k = 0; k < features_count; k++)
                    le_matrix_set_element(self->x, k, j, le_matrix_at(x_train, k, i));
                j++;
            }
        }
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

LeMatrix *
le_svm_predict(LeSVM *self, LeMatrix *x)
{
    if (self == NULL)
        return NULL;
    
    /* In case we use linear kernel and have weights, apply linear classification */
    if (self->weights != NULL)
    {
        LeMatrix *weights_transposed = le_matrix_new_transpose(self->weights);
        LeMatrix *y_predicted = le_matrix_new_product(weights_transposed, x);
        le_matrix_free(weights_transposed);
        le_matrix_add_scalar(y_predicted, self->bias);
        le_matrix_apply_svm_prediction(y_predicted);
        return y_predicted;
    }
    else
    {
        if (self->alphas == NULL)
            return NULL;
        
        unsigned test_examples_count = le_matrix_get_width(x);
        LeMatrix *y_predicted = le_matrix_new_uninitialized(1, test_examples_count);
        for (unsigned i = 0; i < test_examples_count; i++)
        {
            LeMatrix *example = le_matrix_get_column(x, i);
            
            unsigned j;
            float margin = 0;
            unsigned training_examples_count = le_matrix_get_width(self->x);
            for (j = 0; j < training_examples_count; j++)
            {
                margin += le_matrix_at(self->alphas, 0, j) * le_matrix_at(self->y, 0, j) * le_svm_kernel(self->x, example, self->kernel);
            }
            margin += self->bias;
            
            le_matrix_set_element(y_predicted, 0, i, margin);
            le_matrix_free(example);
        }
        le_matrix_apply_svm_prediction(y_predicted);
        return y_predicted;
    }
}

void
le_svm_free(LeSVM *self)
{
    free(self);
}
