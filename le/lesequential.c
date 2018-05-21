/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lesequential.h"
#include <stdlib.h>

struct LeSequential
{
    
};

LeSequential *
le_sequential_new(void)
{
    LeSequential *self = malloc(sizeof(struct LeSequential));
    return self;
}

LeMatrix *
le_sequential_predict(LeSequential *self, LeMatrix *x)
{
    LeMatrix *y = NULL;
    return y;
}

void le_sequential_free(LeSequential *self)
{
    free(self);
}
