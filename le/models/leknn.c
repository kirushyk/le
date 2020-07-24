/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leknn.h"
#include "lemodel.h"
#include <le/tensors/letensor-imp.h>
#include <le/tensors/letensor.h>

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

void                    
le_knn_train(LeKNN *self, LeTensor *x, LeTensor *y, unsigned k)
{
    self->x = x;
    self->y = y;
    self->k = k;
}

void                    
le_knn_free(LeKNN *self)
{
    le_tensor_free(self->x);
    le_tensor_free(self->y);
}
