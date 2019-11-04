/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leshape.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdarg.h>

LeShape *
le_shape_new(unsigned num_dimensions, ...)
{
    LeShape *self = malloc(sizeof(LeShape));
    self->num_dimensions = num_dimensions;
    
    va_list args;
    va_start(args, num_dimensions);
    
    for (uint8_t i = 0; i < num_dimensions; i++)
    {
        int size = va_arg(args, int);
        self->sizes[i] = size;
    }
    
    va_end(args);
    
    return self;
}

LeShape *
le_shape_copy(LeShape *another)
{
    LeShape *self = malloc(sizeof(LeShape));
    self->num_dimensions = another->num_dimensions;
    size_t size = self->num_dimensions * sizeof(uint32_t);
    self->sizes = malloc(size);
    memcpy(self->sizes, another->sizes, size);
    return self;
}

void
le_shape_free(LeShape *self)
{
    if (self)
    {
        free(self->sizes);
        free(self);
    }
}
