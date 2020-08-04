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

struct LeKNN
{
    LeModel parent;
    unsigned k;
    LeTensor *x;
    LeTensor *y;
};

typedef struct LeKNNClass
{
    LeModelClass parent;
} LeKNNClass;

LeKNNClass *
le_knn_class_ensure_init(void)
{
    static bool initialized = false;
    static LeKNNClass klass;
    
    if (!initialized)
    {
        klass.parent.predict =
            (LeTensor *(*)(LeModel *, const LeTensor *))le_knn_predict;
        klass.parent.get_gradients = NULL;
        initialized = 1;
    }

    return &klass;
}

void
le_knn_construct(LeKNN *self)
{
    le_model_construct(LE_MODEL(self));
    LE_OBJECT_GET_CLASS(self) = LE_CLASS(le_knn_class_ensure_init());
    self->x = NULL;
    self->y= NULL;
    self->k = 1;
}

LeKNN *
le_knn_new(void)
{
    LeKNN *self = malloc(sizeof(struct LeKNN));
    le_knn_construct(self);
    return self;
}

void                    
le_knn_train(LeKNN *self, LeTensor *x, LeTensor *y, unsigned k)
{
    assert(x);
    assert(y);
    assert(k > 0);
    self->x = x;
    self->y = y;
    unsigned examples_count = le_matrix_get_width(x);
    assert(examples_count == le_matrix_get_width(y));
    assert(examples_count >= k);
    self->k = k;
}

LeTensor *
le_knn_predict(LeKNN *self, const LeTensor *x)
{
    unsigned train_examples_count = le_matrix_get_width(self->x);
    unsigned test_examples_count = le_matrix_get_width(x);
    unsigned features_count = le_matrix_get_height(x);
    assert(le_matrix_get_height(self->x) == features_count);
    LeTensor *h = le_matrix_new_uninitialized(LE_TYPE_FLOAT32, 1, test_examples_count);
    float *squared_distances = (float *)malloc(train_examples_count * sizeof(float));
    unsigned *indices = (unsigned *)malloc(self->k * sizeof(unsigned));
    for (unsigned i = 0; i < test_examples_count; i++)
    {
        for (unsigned j = 0; j < train_examples_count; j++)
        {
            float squared_distance = 0.0f;
            for (unsigned dim = 0; dim < features_count; dim++)
            {
                float distance = le_matrix_at_f32(self->x, dim, j) - le_matrix_at_f32(x, dim, i);
                squared_distance += distance * distance;
            }
            squared_distances[j] = squared_distance;
        }

        for (unsigned n = 0; n < self->k; n++)
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
        for (unsigned n = 0; n < self->k; n++)
        {
            prediction += le_matrix_at_f32(self->y, 0, indices[n]);
        }
        prediction /= self->k;
        le_matrix_set_f32(h, 0, i, prediction);
    }
    free(indices);
    free(squared_distances);
    return h;
}

void                    
le_knn_free(LeKNN *self)
{
    le_tensor_free(self->x);
    le_tensor_free(self->y);
}
