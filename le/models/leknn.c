/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leknn.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include "lemodel.h"
#include <le/tensors/letensor-imp.h>
#include <le/tensors/letensor.h>
#include <le/tensors/lematrix.h>

typedef struct _LeKNN
{
    LeModel parent;
} LeKNN;

typedef struct _LeKNNPrivate
{
    unsigned k;
    LeTensor *x;
    LeTensor *y;
} LeKNNPrivate;

static void le_knn_class_init (LeKNNClass * klass);
static void le_knn_init (LeKNN * self);
G_DEFINE_FINAL_TYPE_WITH_PRIVATE (LeKNN, le_knn, LE_TYPE_MODEL);

static void
le_knn_dispose (GObject * object)
{
  LeKNN *self = LE_KNN (object);
  g_assert_nonnull (self);
  LeKNNPrivate *priv = le_knn_get_instance_private (self);
  g_assert_nonnull (priv);
  if (priv->x)
    le_tensor_free (priv->x);
  if (priv->y)
    le_tensor_free (priv->y);
  G_OBJECT_CLASS (le_knn_parent_class)->dispose (object);
}

static void
le_knn_finalize (GObject * object)
{
}

LeTensor * le_knn_predict(LeModel * model, const LeTensor *x);

static void
le_knn_class_init (LeKNNClass * klass)
{
  G_OBJECT_CLASS (klass)->dispose = le_knn_dispose;
  G_OBJECT_CLASS (klass)->finalize = le_knn_finalize;
  LE_MODEL_CLASS (klass)->predict = le_knn_predict;
  LE_MODEL_CLASS (klass)->get_gradients = NULL;
}

static void
le_knn_init (LeKNN * self)
{
  LeKNNPrivate *priv = le_knn_get_instance_private (self);
  priv->k = 1;
  priv->x = NULL;
  priv->y = NULL;
}

// LeKNNClass *
// le_knn_class_ensure_init(void)
// {
//     static bool initialized = false;
//     static LeKNNClass klass;
    
//     if (!initialized)
//     {
//         klass.parent.predict =
//             (LeTensor *(*)(LeModel *, const LeTensor *))le_knn_predict;
//         klass.parent.get_gradients = NULL;
//         initialized = 1;
//     }

//     return &klass;
// }

// void
// le_knn_construct(LeKNN *self)
// {
//     le_model_construct(LE_MODEL(self));
//     G_OBJECT_GET_CLASS(self) = G_OBJECT_CLASS(le_knn_class_ensure_init());
//     self->x = NULL;
//     self->y= NULL;
//     self->k = 1;
// }

LeKNN *
le_knn_new (void)
{
  LeKNN *self = g_object_new (le_knn_get_type (), NULL);
  return self;
}

void                    
le_knn_train(LeKNN *self, LeTensor *x, LeTensor *y, unsigned k)
{
    assert(x);
    assert(y);
    assert(k > 0);
    LeKNNPrivate *priv = le_knn_get_instance_private (self);
    priv->x = x;
    priv->y = y;
    unsigned examples_count = le_matrix_get_width(x);
    assert(examples_count == le_matrix_get_width(y));
    assert(examples_count >= k);
    priv->k = k;
}

LeTensor *
le_knn_predict(LeModel * model, const LeTensor *x)
{
  LeKNN *self = LE_KNN (model);
  g_assert_nonnull (self);
  LeKNNPrivate *priv = le_knn_get_instance_private (self);
  g_assert_nonnull (priv);
    unsigned train_examples_count = le_matrix_get_width(priv->x);
    unsigned test_examples_count = le_matrix_get_width(x);
    unsigned features_count = le_matrix_get_height(x);
    assert(le_matrix_get_height(priv->x) == features_count);
    LeTensor *h = le_matrix_new_uninitialized(LE_TYPE_FLOAT32, 1, test_examples_count);
    float *squared_distances = g_new0 (gfloat, train_examples_count);
    unsigned *indices = g_new0(unsigned, priv->k);
    for (unsigned i = 0; i < test_examples_count; i++)
    {
        for (unsigned j = 0; j < train_examples_count; j++)
        {
            float squared_distance = 0.0f;
            for (unsigned dim = 0; dim < features_count; dim++)
            {
                float distance = le_matrix_at_f32(priv->x, dim, j) - le_matrix_at_f32(x, dim, i);
                squared_distance += distance * distance;
            }
            squared_distances[j] = squared_distance;
        }

        for (unsigned n = 0; n < priv->k; n++)
        {
            indices[n] = 0;
            for (unsigned j = 1; j < train_examples_count; j++)
            {
                if (j == indices[n])
                    continue;
                if (squared_distances[j] < squared_distances[indices[n]])
                {
                    indices[n] = j;
                }
            }
            squared_distances[indices[n]] = HUGE_VALF;
        }

        float prediction = 0.0f;
        for (unsigned n = 0; n < priv->k; n++)
        {
            prediction += le_matrix_at_f32(priv->y, 0, indices[n]);
        }
        prediction /= priv->k;
        le_matrix_set_f32(h, 0, i, prediction);
    }
    g_free (indices);
    g_free (squared_distances);
    return h;
}

// void                    
// le_knn_free(LeKNN *self)
// {
//     le_tensor_free(self->x);
//     le_tensor_free(self->y);
// }
