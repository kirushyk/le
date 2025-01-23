/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lelogistic.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <le/tensors/lematrix.h>
#include <le/tensors/letensor-imp.h>
#include "leloss.h"
#include "lemodel.h"
#include "math/lepolynomia.h"

typedef struct _LeLogisticClassifier
{
  LeModel parent;
} LeLogisticClassifier;

typedef struct _LeLogisticClassifierPrivate
{
  LeTensor *weights;
  gfloat bias;
  unsigned polynomia_degree;
} LeLogisticClassifierPrivate;

// typedef struct LeLogisticClassifierClass
// {
//     LeModelClass parent;
// } LeLogisticClassifierClass;
static void le_logistic_classifier_class_init (LeLogisticClassifierClass *klass);
static void le_logistic_classifier_init (LeLogisticClassifier *self);
G_DEFINE_FINAL_TYPE_WITH_PRIVATE (LeLogisticClassifier, le_logistic_classifier, LE_TYPE_MODEL);

static void
le_logistic_classifier_dispose (GObject *object)
{
  LeLogisticClassifier *self = LE_LOGISTIC_CLASSIFIER (object);
  g_assert_nonnull (self);
  LeLogisticClassifierPrivate *priv = le_logistic_classifier_get_instance_private (self);
  g_assert_nonnull (priv);
  if (priv->weights)
    le_tensor_unref (priv->weights);
  G_OBJECT_CLASS (le_logistic_classifier_parent_class)->dispose (object);
}

static void
le_logistic_classifier_finalize (GObject *object)
{
}

LeTensor *le_logistic_classifier_predict (LeModel *model, const LeTensor *x);

static void
le_logistic_classifier_class_init (LeLogisticClassifierClass *klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_logistic_classifier_dispose;
  G_OBJECT_CLASS (klass)->finalize = le_logistic_classifier_finalize;
  LE_MODEL_CLASS (klass)->predict = le_logistic_classifier_predict;
  LE_MODEL_CLASS (klass)->get_gradients = NULL;
}

static void
le_logistic_classifier_init (LeLogisticClassifier *self)
{
}

// LeLogisticClassifierClass *
// le_logistic_classifier_class_ensure_init(void)
// {
//     static bool initialized = false;
//     static LeLogisticClassifierClass klass;

//     if (!initialized)
//     {
//         klass.parent.predict =
//             (LeTensor *(*)(LeModel *, const LeTensor *))le_logistic_classifier_predict;
//         initialized = 1;
//     }

//     return &klass;
// }

// void
// le_logistic_classifier_construct(LeLogisticClassifier *self)
// {
//     le_model_construct(LE_MODEL(self));
//     G_OBJECT_GET_CLASS(self) = G_OBJECT_CLASS(le_logistic_classifier_class_ensure_init());
//     self->weights = NULL;
//     self->bias = 0;
//     self->polynomia_degree = 0;
// }

LeLogisticClassifier *
le_logistic_classifier_new (void)
{
  LeLogisticClassifier *self = g_object_new (le_logistic_classifier_get_type (), NULL);
  return self;
}

LeTensor *
le_logistic_classifier_predict (LeModel *model, const LeTensor *x)
{
  LeLogisticClassifier *self = LE_LOGISTIC_CLASSIFIER (model);
  g_assert_nonnull (self);
  LeLogisticClassifierPrivate *priv = le_logistic_classifier_get_instance_private (self);
  g_assert_nonnull (priv);
  unsigned i;
  LeTensor *wt = le_matrix_new_transpose (priv->weights);
  LeTensor *x_prev = NULL;
  LeTensor *x_poly = NULL;
  for (i = 0; i < priv->polynomia_degree; i++) {
    /// @note: Refrain from init x_prev = x to prevent const
    x_poly = le_matrix_new_polynomia (x_prev ? x_prev : x);
    le_tensor_unref (x_prev);
    x_prev = x_poly;
  }
  LeTensor *a = le_matrix_new_product (wt, x_poly ? x_poly : x);
  le_tensor_unref (wt);
  if (x_poly != x) {
    le_tensor_unref (x_poly);
  }
  le_tensor_add (a, priv->bias);
  le_tensor_apply_sigmoid (a);
  return a;
}

void
le_logistic_classifier_train (LeLogisticClassifier *self, const LeTensor *x_train, const LeTensor *y_train,
    LeLogisticClassifierTrainingOptions options)
{
  unsigned examples_count = le_matrix_get_width (x_train);
  unsigned iterations_count = options.max_iterations;
  unsigned i;

  assert (le_matrix_get_width (y_train) == examples_count);

  LeTensor *x = x_train;
  LeTensor *x_prev = x_train;

  for (i = 0; i < options.polynomia_degree; i++) {
    x = le_matrix_new_polynomia (x_prev);
    if (x_prev != x_train) {
      le_tensor_unref (LE_TENSOR (x_prev));
    }
    x_prev = x;
  }

  unsigned features_count = le_matrix_get_height (x);

  /*
  if (x != x_train)
  {
      le_tensor_unref(x);
  }
  */

  LeLogisticClassifierPrivate *priv = le_logistic_classifier_get_instance_private (self);
  g_assert_nonnull (priv);
  priv->weights = le_matrix_new_zeros (LE_TYPE_FLOAT32, features_count, 1);
  priv->bias = 0;
  priv->polynomia_degree = options.polynomia_degree;

  for (i = 0; i < iterations_count; i++) {
    printf ("Iteration %u. ", i);

    LeTensor *h = le_logistic_classifier_predict (LE_MODEL (self), x_train);

    gfloat train_set_error = le_logistic_loss (h, y_train);

    le_tensor_sub (h, y_train);
    le_tensor_mul (h, 1.0f / examples_count);
    LeTensor *dwt = le_matrix_new_product_full (h, false, x, true);
    LeTensor *dw = le_matrix_new_transpose (dwt);
    le_tensor_mul (dw, options.learning_rate);
    gfloat db = le_tensor_sum_f32 (h);

    le_tensor_unref (dwt);
    le_tensor_unref (h);
    le_tensor_sub (priv->weights, dw);
    le_tensor_unref (dw);
    priv->bias -= options.learning_rate * db;

    printf ("Train Set Error: %f\n", train_set_error);
  }
}
