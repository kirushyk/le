/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lesequential.h"
#include <stdlib.h>
#include "lelist.h"
#include "lelayer.h"

struct LeSequential
{
    LeList *layers;
};

typedef struct LeSequentialClass
{
    LeModelClass parent;
} LeSequentialClass;

LeSequentialClass le_sequential_class;

static void
le_sequential_class_ensure_init()
{
    static int le_sequential_class_initialized = 0;
    
    if (!le_sequential_class_initialized)
    {
        le_sequential_class.parent.predict =
        (LeTensor *(*)(LeModel *, LeTensor *))le_sequential_predict;
        le_sequential_class_initialized = 1;
    }
}

void
le_sequential_construct(LeSequential *self)
{
    le_model_construct((LeModel *)self);
    le_sequential_class_ensure_init();
    ((LeObject *)self)->klass = (LeClass *)&le_sequential_class;
    
    self->layers = NULL;
}

LeSequential *
le_sequential_new(void)
{
    LeSequential *self = malloc(sizeof(struct LeSequential));
    le_sequential_construct(self);
    return self;
}

LeTensor *
le_sequential_predict(LeSequential *self, LeTensor *x)
{
    LeTensor *signal = x;
    for (LeList *current = self->layers; current != NULL; current = current->next)
    {
        LeLayer *current_layer = (LeLayer *)current->data;
        LeTensor *wx;
        wx = le_matrix_new_product(current_layer->weights, signal);
        le_matrix_apply_sigmoid(wx);
        
    }
    return signal;
}

void
le_sequential_free(LeSequential *self)
{
    free(self);
}
