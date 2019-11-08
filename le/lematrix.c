/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lematrix.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "letensor-imp.h"

unsigned
le_matrix_get_width(LeTensor *self)
{
    assert(self->shape->num_dimensions == 2);
    
    return self->shape->sizes[1];
}

unsigned
le_matrix_get_height(LeTensor *self)
{
    assert(self->shape->num_dimensions == 2);
    
    return self->shape->sizes[0];
}

float
le_matrix_at(LeTensor *self, unsigned y, unsigned x)
{
    assert(self->shape->num_dimensions == 2);
    
    assert(y < self->shape->sizes[0]);
    assert(x < self->shape->sizes[1]);
    
    return self->data[y * self->shape->sizes[1] + x];
}

void
le_matrix_set_element(LeTensor *self, unsigned y, unsigned x, float value)
{
    assert(self->shape->num_dimensions == 2);
    
    assert(y < self->shape->sizes[0]);
    assert(x < self->shape->sizes[1]);
    
    self->data[y * self->shape->sizes[1] + x] = value;
}

LeTensor *
le_matrix_new_from_data(unsigned height, unsigned width, const float *data)
{
    LeTensor *self;
    size_t data_size = height * width * sizeof(float);
    
    self = malloc(sizeof(struct LeTensor));
    self->data = malloc(data_size);
    self->shape = le_shape_new(2, height, width);
    self->element_type = LE_TYPE_FLOAT32;
    memcpy(self->data, data, data_size);
    
    return self;
}

LeTensor *
le_matrix_new_identity(unsigned size)
{
    unsigned x;
    unsigned y;
    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->data = malloc(size * size * sizeof(float));
    self->shape = le_shape_new(2, size, size);
    self->element_type = LE_TYPE_FLOAT32;
    
    for (y = 0; y < size; y++)
    {
        for (x = 0; x < size; x++)
        {
            self->data[y * size + x] = (x == y) ? 1.0 : 0.0;
        }
    }
    
    return self;
}

LeTensor *
le_matrix_new_uninitialized(unsigned height, unsigned width)
{
    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->data = malloc(height * width * sizeof(float));
    self->shape = le_shape_new(2, height, width);
    self->element_type = LE_TYPE_FLOAT32;
    
    return self;
}

LeTensor *
le_matrix_new_zeros(unsigned height, unsigned width)
{
    unsigned i;
    unsigned elements_count;
    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->data = malloc(height * width * sizeof(float));
    self->shape = le_shape_new(2, height, width);
    self->element_type = LE_TYPE_FLOAT32;
    elements_count = height * width;
    
    for (i = 0; i < elements_count; i++)
    {
        self->data[i] = 0.0f;
    }
    
    return self;
}


LeTensor *
le_matrix_new_rand(unsigned height, unsigned width)
{
    unsigned x;
    unsigned y;
    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->data = malloc(height * width * sizeof(float));
    self->shape = le_shape_new(2, height, width);
    self->element_type = LE_TYPE_FLOAT32;
    
    for (y = 0; y < self->shape->sizes[0]; y++)
    {
        for (x = 0; x < self->shape->sizes[1]; x++)
        {
            self->data[y * self->shape->sizes[1] + x] = rand() / (float)RAND_MAX;
        }
    }
    
    return self;
}

LeTensor *
le_matrix_new_transpose(LeTensor *a)
{
    assert(a->shape->num_dimensions == 2);

    unsigned x;
    unsigned y;
    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->data = malloc(a->shape->sizes[1] * a->shape->sizes[0] * sizeof(float));
    self->shape = le_shape_new(2, a->shape->sizes[1], a->shape->sizes[0]);
    self->element_type = a->element_type;
    
    for (y = 0; y < self->shape->sizes[0]; y++)
    {
        for (x = 0; x < self->shape->sizes[1]; x++)
        {
            self->data[y * self->shape->sizes[1] + x] = a->data[x * a->shape->sizes[1] + y];
        }
    }
    
    return self;
}

LeTensor *
le_matrix_new_product(LeTensor *a, LeTensor *b)
{
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);

    unsigned x;
    unsigned y;
    unsigned i;
    
    LeTensor *self;
    
    if (a->shape->sizes[1] != b->shape->sizes[0])
        return le_tensor_new();
        
    if (a->element_type != b->element_type)
        return le_tensor_new();
    
    self = malloc(sizeof(struct LeTensor));
    self->shape = le_shape_new(2, a->shape->sizes[0], b->shape->sizes[1]);
    self->data = malloc(le_shape_get_elements_count(self->shape) * sizeof(float));
    self->element_type = a->element_type;
    
    for (y = 0; y < self->shape->sizes[0]; y++)
    {
        for (x = 0; x < self->shape->sizes[1]; x++)
        {
            self->data[y * self->shape->sizes[1] + x] = 0.0f;
            for (i = 0; i < a->shape->sizes[1]; i++)
            {
                self->data[y * self->shape->sizes[1] + x] += a->data[y * a->shape->sizes[1] + i] * b->data[i * b->shape->sizes[1] + x];
            }
        }
    }
    
    return self;
}

LeTensor *
le_matrix_get_column(LeTensor *self, unsigned x)
{
    assert(self->shape->num_dimensions == 2);
    /// @todo: Add dimension checks
    
    unsigned y;
    unsigned height = le_matrix_get_height(self);
    LeTensor *column = le_matrix_new_uninitialized(height, 1);
    
    for (y = 0; y < height; y++)
    {
        column->data[y] = self->data[y * self->shape->sizes[1] + x];
    }
    
    return column;
}
