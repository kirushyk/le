/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leshape.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>
#include <stdio.h>

LeShape *
le_shape_new(unsigned num_dimensions, ...)
{
    LeShape *self = le_shape_new_uninitialized(num_dimensions);
    
    va_list args;
    va_start(args, num_dimensions);
    
    for (unsigned i = 0; i < num_dimensions; i++)
    {
        int size = va_arg(args, int);
        self->sizes[i] = size;
    }
    
    va_end(args);
    
    return self;
}

LeShape *
le_shape_new_uninitialized(unsigned num_dimensions)
{
    LeShape *self = malloc(sizeof(LeShape));
    self->num_dimensions = num_dimensions;
    self->sizes = malloc(num_dimensions * sizeof(uint32_t));
    
    return self;
}

uint32_t *
le_shape_get_data(LeShape *shape)
{
    return shape->sizes;
}

uint32_t
le_shape_get_size(LeShape *shape, int dimension)
{
    assert(shape);
    assert(dimension < (int)shape->num_dimensions);
    assert(dimension >= -(int)shape->num_dimensions);
    return shape->sizes[(dimension < 0) ? (shape->num_dimensions + dimension) : dimension];
}

void
le_shape_set_size(LeShape *shape, unsigned dimension, uint32_t size)
{
    assert(shape);
    assert(dimension < shape->num_dimensions);
    shape->sizes[dimension] = size;
}

LeShape *
le_shape_copy(LeShape *another)
{
    assert(another);
    LeShape *self = malloc(sizeof(LeShape));
    self->num_dimensions = another->num_dimensions;
    size_t size = self->num_dimensions * sizeof(uint32_t);
    self->sizes = malloc(size);
    memcpy(self->sizes, another->sizes, size);
    return self;
}

LeShape *
le_shape_lower_dimension(LeShape *another)
{
    /// @todo: Add assertions
    LeShape *self = malloc(sizeof(LeShape));
    self->num_dimensions = another->num_dimensions - 1;
    size_t size = self->num_dimensions * sizeof(uint32_t);
    self->sizes = malloc(size);
    memcpy(self->sizes, another->sizes + 1, size);
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

const char *
le_shape_to_cstr(LeShape *shape)
{
    static char buffer[1024];

    if (shape)
    {
        char *ptr = buffer;
        ptr[0] = '(';
        /// @todo: Add overflow check
        ptr++;
        for (unsigned i = 0; i < shape->num_dimensions; i++)
        {
            int written = 0;
            if (shape->sizes[i])
            {
                sprintf(ptr, "%d%n", shape->sizes[i], &written);
            }
            else
            {
                sprintf(ptr, "?");
                written = 1;
            }
            ptr += written;
            
            sprintf(ptr, "%s%n",
                    i == (shape->num_dimensions - 1) ? ")" : ", ",
                    &written);
            ptr += written;
        }
    }
    else
    {
        sprintf(buffer, "(null)");
    }

    return buffer;
}

uint32_t
le_shape_get_elements_count(LeShape *shape)
{
    assert(shape);
    assert(shape->sizes);
    
    uint32_t count = 0;
    if (shape)
    {
        count = 1;
        for (unsigned i = 0; i < shape->num_dimensions; i++)
        {
            count *= shape->sizes[i];
        }
    }
    return count;
}

uint32_t
le_shape_get_regions_count(LeShape *shape)
{
    assert(shape);
    assert(shape->sizes);
    
    uint32_t count = 0;
    if (shape)
    {
        count = 1;
        for (unsigned i = 0; i < shape->num_dimensions - 1; i++)
        {
            count *= shape->sizes[i];
        }
    }
    return count;
}

bool
le_shape_equal(LeShape *a, LeShape *b)
{
    assert(a);
    assert(b);
    
    if (a->num_dimensions != b->num_dimensions)
        return false;
    
    for (unsigned i = 0; i < a->num_dimensions; i++)
    {
        if (a->sizes[i] != b->sizes[i])
            return false;
    }
    
    return true;
}
