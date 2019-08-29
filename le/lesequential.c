/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lesequential.h"
#include <stdlib.h>

struct LeSequential
{
    
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
        (LeMatrix *(*)(LeModel *, LeMatrix *))le_sequential_predict;
        le_sequential_class_initialized = 1;
    }
}

void
le_sequential_construct(LeSequential *self)
{
    le_model_construct((LeModel *)self);
    le_sequential_class_ensure_init();
    ((LeObject *)self)->klass = (LeClass *)&le_sequential_class;
}

LeSequential *
le_sequential_new(void)
{
    LeSequential *self = malloc(sizeof(struct LeSequential));
    le_sequential_construct(self);
    return self;
}

LeMatrix *
le_sequential_predict(LeSequential *self, LeMatrix *x)
{
    LeMatrix *y = NULL;
    return y;
}

void
le_sequential_free(LeSequential *self)
{
    free(self);
}
