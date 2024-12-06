/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lesvm.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "lemodel.h"
#include <le/tensors/lematrix.h>

typedef struct _LeSVM
{
    LeModel   parent;
} LeSVM;

typedef struct _LeSVMPrivate
{    
    /* Training data */
    LeTensor *x;
    LeTensor *y;
    
    LeKernel  kernel;
    gfloat     bias;
    /* Weights for linear classifier */
    LeTensor *weights;
    LeTensor *alphas;
} LeSVMPrivate;

// typedef struct LeSVMClass
// {
//     LeModelClass parent;
// } LeSVMClass;
static void le_svm_class_init (LeSVMClass * klass);
static void le_svm_init (LeSVM * self);
G_DEFINE_FINAL_TYPE_WITH_PRIVATE (LeSVM, le_svm, LE_TYPE_MODEL);

static void
le_svm_dispose (GObject * object)
{
  LeSVM *self = LE_SVM (object);
  g_assert_nonnull (self);
  LeSVMPrivate *priv = le_svm_get_instance_private (self);
  g_assert_nonnull (priv);
  if (priv->x)
    le_tensor_free (priv->x);
  if (priv->y)
    le_tensor_free (priv->y);
  if (priv->weights)
    le_tensor_free (priv->weights);
  if (priv->alphas)
    le_tensor_free (priv->alphas);
  G_OBJECT_CLASS (le_svm_parent_class)->dispose (object);
}

static void
le_svm_finalize (GObject * object)
{
}


LeTensor * le_svm_predict(LeModel *self, const LeTensor *x);

static void
le_svm_class_init (LeSVMClass * klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_svm_dispose;
  G_OBJECT_CLASS (klass)->finalize = le_svm_finalize;
  LE_MODEL_CLASS (klass)->predict = le_svm_predict;
  LE_MODEL_CLASS (klass)->get_gradients = NULL;
}

static void
le_svm_init (LeSVM * self)
{
  LeSVMPrivate *priv = le_svm_get_instance_private (self);
  g_assert_nonnull (priv);
  priv->bias = 0.0f;
  priv->x = NULL;
  priv->y = NULL;
  priv->alphas = NULL;
  priv->weights = NULL;
  priv->kernel = LE_KERNEL_LINEAR;
}

LeSVM *
le_svm_new (void)
{
  LeSVM *self = g_object_new (le_svm_get_type (), NULL);
  return self;
}

static gfloat
kernel_function(const LeTensor *a, const LeTensor *b, LeKernel kernel)
{
    switch (kernel) {
    case LE_KERNEL_RBF:
        return le_rbf(a, b, 0.5f);
    case LE_KERNEL_LINEAR:
    default:
        return le_dot_product(a, b);
    }
}

LeTensor *
le_svm_margins(LeSVM *self, const LeTensor *x)
{
  g_assert_nonnull (self);
  LeSVMPrivate *priv = le_svm_get_instance_private (self);
  g_assert_nonnull (priv);
    
    /* In case we use linear kernel and have weights, apply linear classification */
  if (priv->weights != NULL)
  {
    LeTensor *weights_transposed = le_matrix_new_transpose (priv->weights);
    LeTensor *margins = le_matrix_new_product (weights_transposed, x);
    le_tensor_free (weights_transposed);
    le_tensor_add (margins, priv->bias);
    return margins;
  }
  else
  {
    if (priv->alphas == NULL)
      return NULL;
      
    unsigned test_examples_count = le_matrix_get_width (x);
    LeTensor *margins = le_matrix_new_uninitialized (LE_TYPE_FLOAT32, 1, test_examples_count);
    for (unsigned i = 0; i < test_examples_count; i++)
    {
      LeTensor *example = le_matrix_get_column (x, i);
      
      unsigned j;
      gfloat margin = 0;
      unsigned training_examples_count = le_matrix_get_width (priv->x);
      for (j = 0; j < training_examples_count; j++)
      {
          LeTensor *x_train_j = le_matrix_get_column (priv->x, j);
          gfloat alphaj = le_matrix_at_f32 (priv->alphas, 0, j);
          if (alphaj > 1e-4f || alphaj < -1e-4f)
          {
              margin += alphaj * le_matrix_at_f32 (priv->y, 0, j) * kernel_function (x_train_j, example, priv->kernel);
          }
          le_tensor_free (x_train_j);
      }
      margin += priv->bias;
      
      le_matrix_set (margins, 0, i, margin);
      le_tensor_free (example);
    }
    return margins;
  }
}

void
le_svm_train (LeSVM * self, const LeTensor * x_train, const LeTensor * y_train, LeSVMTrainingOptions options)
{
  g_assert_nonnull (self);
  LeSVMPrivate *priv = le_svm_get_instance_private (self);
  g_assert_nonnull (priv);
  unsigned passes = 0;
  /// @todo: Expose this parameter
  unsigned max_passes = 100;
  unsigned max_iterations = 10000;
  
  unsigned features_count = le_matrix_get_height (x_train);
  unsigned examples_count = le_matrix_get_width (x_train);
  /// @todo: Add more clever input data checks
  assert(examples_count == le_matrix_get_width (y_train));

  /// @todo: Add checks
  if (priv->x)
    le_tensor_free (priv->x); 
  priv->x = le_tensor_new_copy ((LeTensor *)x_train);
  if (priv->y)
    le_tensor_free (priv->y);
  priv->y = le_tensor_new_copy ((LeTensor *)y_train);
  priv->kernel = options.kernel;
  /// @todo: Add cleanup here
  /// @note: Maybe use stack variable instead
  priv->alphas = le_matrix_new_zeros (LE_TYPE_FLOAT32, 1, examples_count);
  priv->bias = 0;
  /// @todo: Add cleanup here
  priv->weights = NULL;
    
  const gfloat tol = 1e-4f;
  const gfloat C = options.c;

  /// @note: Sequential Minimal Optimization (SMO) algorithm
  for (unsigned iteration = 0; passes < max_passes && iteration < max_iterations; iteration++)
  {
      unsigned num_changed_alphas = 0;
      
      for (int i = 0; i < examples_count; i++)
      {
          /// @todo: Implement immutable matrix columns
          LeTensor *x_train_i = le_matrix_get_column(x_train, i);
          /// @note: We will have 1x1 matrix here
          LeTensor *shallow_margin_matrix = le_svm_margins(self, x_train_i);
          gfloat margin = le_matrix_at_f32(shallow_margin_matrix, 0, 0);
          le_tensor_free(shallow_margin_matrix);
          gfloat Ei = margin - le_matrix_at_f32(y_train, 0, i);
          if ((le_matrix_at_f32(y_train, 0, i) * Ei < -tol && le_matrix_at_f32(priv->alphas, 0, i) < C) ||
              (le_matrix_at_f32(y_train, 0, i) * Ei > tol && le_matrix_at_f32(priv->alphas, 0, i) > 0.0f))
          {
              int j = i;
              while (j == i)
                  j = rand() % examples_count;
              /// @todo: Implement immutable matrix columns
              LeTensor *x_train_j = le_matrix_get_column(x_train, j);
              /// @note: We will have 1x1 matrix here
              LeTensor *shallow_margin_matrix = le_svm_margins(self, x_train_j);
              gfloat margin = le_matrix_at_f32(shallow_margin_matrix, 0, 0);
              le_tensor_free(shallow_margin_matrix);
              gfloat Ej = margin - le_matrix_at_f32(y_train, 0, j);
              
              gfloat ai = le_matrix_at_f32(priv->alphas, 0, i);
              gfloat aj = le_matrix_at_f32(priv->alphas, 0, j);
              gfloat L = 0, H = C;
              if (le_matrix_at_f32(y_train, 0, i) == le_matrix_at_f32(y_train, 0, j))
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
                  gfloat eta = 2 * kernel_function(x_train_i, x_train_j, priv->kernel) -
                      kernel_function(x_train_i, x_train_i, priv->kernel) -
                      kernel_function(x_train_j, x_train_j, priv->kernel);
                  if (eta < 0)
                  {
                      gfloat newaj = aj - le_matrix_at_f32(y_train, 0, j) * (Ei - Ej) / eta;
                      if (newaj > H)
                          newaj = H;
                      if (newaj < L)
                          newaj = L;
                      if (fabs(aj - newaj) >= 1e-4)
                      {
                          le_matrix_set(priv->alphas, 0, j, newaj);
                          gfloat newai = ai + le_matrix_at_f32(y_train, 0, i) * le_matrix_at_f32(y_train, 0, j) * (aj - newaj);
                          le_matrix_set(priv->alphas, 0, i, newai);
                          
                          gfloat b1 = priv->bias - Ei - le_matrix_at_f32(y_train, 0, i) * (newai - ai) * kernel_function(x_train_i, x_train_i, priv->kernel)
                          - le_matrix_at_f32(y_train, 0, j) * (newaj - aj) * kernel_function(x_train_i, x_train_j, priv->kernel);
                          gfloat b2 = priv->bias - Ej - le_matrix_at_f32(y_train, 0, i) * (newai - ai) * kernel_function(x_train_i, x_train_j, priv->kernel)
                          - le_matrix_at_f32(y_train, 0, j) * (newaj - aj) * kernel_function(x_train_j, x_train_j, priv->kernel);
                          priv->bias = 0.5f * (b1 + b2);
                          if (newai > 0 && newai < C)
                              priv->bias = b1;
                          if (newaj > 0 && newaj < C)
                              priv->bias = b2;
                          
                          num_changed_alphas++;
                      }
                  }
              }
              le_tensor_free(x_train_j);
          }
          le_tensor_free(x_train_i);
      }

      if (num_changed_alphas == 0)
          passes++;
      else
          passes = 0;
  }
  
  if (priv->kernel == LE_KERNEL_LINEAR)
  {
      /* For linear kernel, we calculate weights */
      priv->weights = le_matrix_new_uninitialized(LE_TYPE_FLOAT32, features_count, 1);
      for (int j = 0; j < features_count; j++)
      {
          gfloat s = 0.0f;
          for (int i = 0; i < examples_count; i++)
          {
              s += le_matrix_at_f32(priv->alphas, 0, i) * le_matrix_at_f32(y_train, 0, i) * le_matrix_at_f32(x_train, j, i);
          }
          le_matrix_set(priv->weights, j, 0, s);
      }
  }
  else
  {
      /* For other kernels, we only retain alphas and training data for support vectors */
      unsigned support_vectors_count = 0;
      const gfloat alpha_tolerance = 1e-4f;
      for (int i = 0; i < examples_count; i++)
      {
          if (le_matrix_at_f32(priv->alphas, 0, i) >= alpha_tolerance)
              support_vectors_count++;
      }
      
      LeTensor *new_alphas = le_matrix_new_uninitialized(LE_TYPE_FLOAT32, 1, support_vectors_count);
      priv->x = le_matrix_new_uninitialized(LE_TYPE_FLOAT32, features_count, support_vectors_count);
      priv->y = le_matrix_new_uninitialized(LE_TYPE_FLOAT32, 1, support_vectors_count);

      int j = 0; /// Iterator for new matrices
      for (int i = 0; i < examples_count; i++)
      {
          if (le_matrix_at_f32(priv->alphas, 0, i) >= alpha_tolerance)
          {
              le_matrix_set(new_alphas, 0, j, le_matrix_at_f32(priv->alphas, 0, i));
              le_matrix_set(priv->y, 0, j, le_matrix_at_f32(y_train, 0, i));
              for (int k = 0; k < features_count; k++)
                  le_matrix_set(priv->x, k, j, le_matrix_at_f32(x_train, k, i));
              j++;
          }
      }
      
      le_tensor_free(priv->alphas);
      priv->alphas = new_alphas;
  }
}

LeTensor *
le_svm_predict (LeModel * model, const LeTensor * x)
{
  g_assert_nonnull (model);
  LeSVM *self = LE_SVM (model);
  g_assert_nonnull (self);
  g_assert_nonnull (x);

  LeTensor *y_predicted = le_svm_margins (self, x);
  le_tensor_apply_sgn (y_predicted);
  return y_predicted;
}
