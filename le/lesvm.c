/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lesvm.h"
#include <math.h>
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

static float
kernel_function(LeMatrix *a, LeMatrix *b, LeKernel kernel)
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
le_svm_margins(LeSVM *self, LeMatrix *x)
{
    if (self == NULL)
        return NULL;
    
    /* In case we use linear kernel and have weights, apply linear classification */
    if (self->weights != NULL)
    {
        LeMatrix *weights_transposed = le_matrix_new_transpose(self->weights);
        LeMatrix *margins = le_matrix_new_product(weights_transposed, x);
        le_matrix_free(weights_transposed);
        le_matrix_add_scalar(margins, self->bias);
        return margins;
    }
    else
    {
        if (self->alphas == NULL)
            return NULL;
        
        unsigned test_examples_count = le_matrix_get_width(x);
        LeMatrix *margins = le_matrix_new_uninitialized(1, test_examples_count);
        for (unsigned i = 0; i < test_examples_count; i++)
        {
            LeMatrix *example = le_matrix_get_column(x, i);
            
            unsigned j;
            float margin = 0;
            unsigned training_examples_count = le_matrix_get_width(self->x);
            for (j = 0; j < training_examples_count; j++)
            {
                margin += le_matrix_at(self->alphas, 0, j) * le_matrix_at(self->y, 0, j) * kernel_function(self->x, example, self->kernel);
            }
            margin += self->bias;
            
            le_matrix_set_element(margins, 0, i, margin);
            le_matrix_free(example);
        }
        return margins;
    }
}

void
le_svm_train(LeSVM *self, LeMatrix *x_train, LeMatrix *y_train, LeKernel kernel)
{
    unsigned passes = 0;
    /// @todo: Expose this parameter
    unsigned max_passes = 10;
    unsigned max_iterations = 10000;
    
    unsigned features_count = le_matrix_get_height(x_train);
    unsigned examples_count = le_matrix_get_width(x_train);
    /// @todo: Add more clever input data checks
    assert(examples_count == le_matrix_get_width(y_train));

    /// @todo: Add checks
    self->x = x_train;
    self->y = y_train;
    self->kernel = kernel;
    /// @todo: Add cleanup here
    /// @note: Maybe use stack variable instead
    self->alphas = le_matrix_new_zeros(1, examples_count);
    self->bias = 0;
    /// @todo: Add cleanup here
    self->weights = NULL;
    
    const float tol = 1e-4f;
    const float C = 1.0f;

    /// @note: Sequential Minimal Optimization (SMO) algorithm
    for (unsigned iteration = 0; passes < max_passes && iteration < max_iterations; iteration++)
    {
        unsigned num_changed_alphas = 0;
        
        for (int i = 0; i < examples_count; i++)
        {
            /// @todo: Implement immutable matrix columns
            LeMatrix *x_train_i = le_matrix_get_column(x_train, i);
            /// @note: We will have 1x1 matrix here
            LeMatrix *shallow_margin_matrix = le_svm_margins(self, x_train_i);
            float margin = le_matrix_at(shallow_margin_matrix, 0, 0);
            le_matrix_free(shallow_margin_matrix);
            float Ei = margin - le_matrix_at(y_train, 0, i);
            if ((le_matrix_at(y_train, 0, i) * Ei < -tol && le_matrix_at(self->alphas, 0, i) < C) ||
                (le_matrix_at(y_train, 0, i) * Ei > tol && le_matrix_at(self->alphas, 0, i) > 0.0f))
            {
                int j = i;
                while (j == i)
                    j = rand() % examples_count;
                /// @todo: Implement immutable matrix columns
                LeMatrix *x_train_j = le_matrix_get_column(x_train, j);
                /// @note: We will have 1x1 matrix here
                LeMatrix *shallow_margin_matrix = le_svm_margins(self, x_train_j);
                float margin = le_matrix_at(shallow_margin_matrix, 0, 0);
                le_matrix_free(shallow_margin_matrix);
                float Ej = margin - le_matrix_at(y_train, 0, j);
                
                float ai = le_matrix_at(self->alphas, 0, i);
                float aj = le_matrix_at(self->alphas, 0, j);
                float L = 0, H = C;
                if (le_matrix_at(y_train, 0, i) == le_matrix_at(y_train, 0, j))
                {
                    L = fmax(0, ai + aj - C);
                    H = fmin(C, ai + aj);
                }
                else
                {
                    L = fmax(0, aj - ai);
                    H = fmin(C, C + aj - ai);
                }
                
                if (fabs(L - H) > 1e-4f)
                {
                    float eta = 2 * kernel_function(x_train_i, x_train_j, kernel) -
                        kernel_function(x_train_i, x_train_i, kernel) -
                        kernel_function(x_train_j, x_train_j, kernel);
                    if (eta < 0)
                    {
                        // compute new alpha_j and clip it inside [0 C]x[0 C] box
                        // then compute alpha_i based on it.
                        float newaj = aj - le_matrix_at(y_train, 0, j) * (Ei - Ej) / eta;
                        if (newaj > H)
                            newaj = H;
                        if (newaj < L)
                            newaj = L;
                        if (fabs(aj - newaj) >= 1e-4)
                        {
                            le_matrix_set_element(self->alphas, j, 0, newaj);
                            float newai = ai + le_matrix_at(y_train, 0, i) * le_matrix_at(y_train, 0, j) * (aj - newaj);
                            le_matrix_set_element(self->alphas, i, 0, newai);
                            
                            float b1 = self->bias - Ei - le_matrix_at(y_train, 0, i) * (newai - ai) * kernel_function(x_train_i, x_train_i, kernel)
                            - le_matrix_at(y_train, 0, j) * (newaj - aj) * kernel_function(x_train_i, x_train_j, kernel);
                            float b2 = self->bias - Ej - le_matrix_at(y_train, 0, i) * (newai - ai) * kernel_function(x_train_i, x_train_j, kernel)
                            - le_matrix_at(y_train, 0, j) * (newaj - aj) * kernel_function(x_train_j, x_train_j, kernel);
                            self->bias = 0.5f * (b1 + b2);
                            if (newai > 0 && newai < C)
                                self->bias = b1;
                            if (newaj > 0 && newaj < C)
                                self->bias = b2;
                            
                            num_changed_alphas++;
                        }
                    }
                }
                le_matrix_free(x_train_j);
            }
            le_matrix_free(x_train_i);
        }

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
                s += le_matrix_at(self->alphas, 0, i) * le_matrix_at(y_train, 0, i) * le_matrix_at(x_train, j, i);
            }
            le_matrix_set_element(self->weights, j, 0, s);
        }
    }
    else
    {
        /* For other kernels, we only retain alphas and training data for support vectors */
        unsigned support_vectors_count = 0;
        const float alpha_tolerance = 1e-4f;
        for (int i = 0; i < examples_count; i++)
        {
            if (le_matrix_at(self->alphas, 0, i) >= alpha_tolerance)
                support_vectors_count++;
        }
        
        LeMatrix *new_alphas = le_matrix_new_uninitialized(1, support_vectors_count);
        self->x = le_matrix_new_uninitialized(features_count, support_vectors_count);
        self->y = le_matrix_new_uninitialized(1, support_vectors_count);

        int j = 0; /// Iterator for new matrices
        for (int i = 0; i < examples_count; i++)
        {
            if (le_matrix_at(self->alphas, 0, i) >= alpha_tolerance)
            {
                le_matrix_set_element(new_alphas, 0, j, le_matrix_at(self->alphas, 0, i));
                le_matrix_set_element(self->y, 0, j, le_matrix_at(y_train, 0, i));
                for (int k = 0; k < features_count; k++)
                    le_matrix_set_element(self->x, k, j, le_matrix_at(x_train, k, i));
                j++;
            }
        }
        
        le_matrix_free(self->alphas);
        self->alphas = new_alphas;
    }
}

LeMatrix *
le_svm_predict(LeSVM *self, LeMatrix *x)
{
    assert(self != NULL);
    assert(x != NULL);

    LeMatrix *y_predicted = le_svm_margins(self, x);
    le_matrix_apply_svm_prediction(y_predicted);
    return y_predicted;
}

void
le_svm_free(LeSVM *self)
{
    free(self);
}
