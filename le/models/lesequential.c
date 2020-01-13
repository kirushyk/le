/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lesequential.h"
#include <assert.h>
#include <stdlib.h>
#include "lelist.h"
#include "lematrix.h"

struct LeSequential
{
    LeModel parent;
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

void
le_sequential_add(LeSequential *self, LeLayer *layer)
{
    self->layers = le_list_append(self->layers, layer);
    LeList *parameters = le_layer_get_parameters(layer);
    
    for (LeList *current = parameters; current != NULL; current = current->next)
    {
        LeTensor *parameter = (LeTensor *)current->data;
        le_model_append_parameter(LE_MODEL(self), parameter);
    }
}

LeTensor *
le_sequential_predict(LeSequential *self, LeTensor *x)
{
    assert(self);
    assert(x);
    
    LeTensor *signal = le_tensor_new_copy(x);
    for (LeList *current = self->layers; current != NULL; current = current->next)
    {
        LeLayer *current_layer = (LeLayer *)current->data;
        printf("signal =\n");
        le_tensor_print(signal, stdout);
        LeTensor *output = le_layer_forward_prop(current_layer, signal);
        le_tensor_free(signal);
        signal = output;
    }
    return signal;
}

void
le_sequential_free(LeSequential *self)
{
    free(self);
}
