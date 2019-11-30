/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lematrix.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "letensor-imp.h"
#ifdef __APPLE__
#include "../platform/accelerate/leaccelerate.h"
#endif

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
    
    return ((float *)self->data)[y * self->shape->sizes[1] + x];
}

void
le_matrix_add(LeTensor *self, LeTensor *another)
{
    unsigned x, y;
    
    assert(self->shape->num_dimensions == 2);
    
    /// @note: Add horizontal broadcasting
    assert(self->shape->sizes[0] == another->shape->sizes[0]);
    assert(another->shape->sizes[1] == 1);
    
    for (y = 0; y < self->shape->sizes[0]; y++)
    {
        for (x = 0; x < self->shape->sizes[1]; x++)
        {
            ((float *)self->data)[y * self->shape->sizes[1] + x] += ((float *)another->data)[y];
        }
    }
}

void
le_matrix_set_element(LeTensor *self, unsigned y, unsigned x, float value)
{
    assert(self->shape->num_dimensions == 2);
    
    assert(y < self->shape->sizes[0]);
    assert(x < self->shape->sizes[1]);
    
    ((float *)self->data)[y * self->shape->sizes[1] + x] = value;
}

LeTensor *
le_matrix_new_from_data(unsigned height, unsigned width, const float *data)
{
    LeTensor *self;
    size_t data_size = height * width * sizeof(float);
    
    self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = le_shape_new(2, height, width);
    self->stride = width;
    self->owns_data = true;
    self->data = malloc(data_size);
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
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = le_shape_new(2, size, size);
    self->stride = size;
    self->owns_data = true;
    self->data = malloc(size * size * sizeof(float));
    
    for (y = 0; y < size; y++)
    {
        for (x = 0; x < size; x++)
        {
            ((float *)self->data)[y * size + x] = (x == y) ? 1.0 : 0.0;
        }
    }
    
    return self;
}

LeTensor *
le_matrix_new_uninitialized(unsigned height, unsigned width)
{
    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = le_shape_new(2, height, width);
    self->stride = width;
    self->owns_data = true;
    self->data = malloc(height * width * sizeof(float));
    
    return self;
}

LeTensor *
le_matrix_new_zeros(unsigned height, unsigned width)
{
    unsigned i;
    unsigned elements_count;
    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->shape = le_shape_new(2, height, width);
    self->element_type = LE_TYPE_FLOAT32;
    self->stride = width;
    self->owns_data = true;
    self->data = malloc(height * width * sizeof(float));
    elements_count = height * width;
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = 0.0f;
    }
    
    return self;
}


LeTensor *
le_matrix_new_rand(unsigned height, unsigned width)
{
    unsigned i;
    unsigned elements_count;
    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = le_shape_new(2, height, width);
    self->stride = width;
    self->owns_data = true;
    self->data = malloc(height * width * sizeof(float));
    elements_count = height * width;
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = rand() / (float)RAND_MAX;
    }
    
    return self;
}

#define TRNASPOSE(type) for (y = 0; y < self->shape->sizes[0]; y++) \
{ \
    for (x = 0; x < self->shape->sizes[1]; x++) \
    { \
        ((type *)self->data)[y * self->shape->sizes[1] + x] = ((type *)a->data)[x * a->shape->sizes[1] + y]; \
    } \
}

LeTensor *
le_matrix_new_transpose(LeTensor *a)
{
    assert(a->shape->num_dimensions == 2);

    unsigned x;
    unsigned y;
    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->element_type = a->element_type;
    self->shape = le_shape_new(2, a->shape->sizes[1], a->shape->sizes[0]);
    self->stride = le_shape_get_last_size(self->shape);
    self->owns_data = true;
    self->data = malloc(le_shape_get_elements_count(self->shape) * le_type_size(self->element_type));
    
    switch (le_type_size(self->element_type))
    {
    case 1:
        TRNASPOSE(uint8_t);
        break;

    case 2:
        TRNASPOSE(uint16_t);
        break;

    case 4:
        TRNASPOSE(uint32_t);
        break;

    case 8:
        TRNASPOSE(uint64_t);
        break;

    default:
        break;
    }
    
    return self;
}

#undef TRANSPOSE


LeTensor *
le_matrix_new_sum(LeTensor *a, unsigned dimension)
{
    unsigned x, y;
    
    assert(a->shape->num_dimensions == 2);

    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = le_shape_new(2, a->shape->sizes[0], 1/*a->shape->sizes[1]*/);
    self->stride = le_shape_get_last_size(self->shape);
    self->owns_data = true;
    self->data = malloc(le_shape_get_elements_count(self->shape) * le_type_size(self->element_type));
    
    assert(/*(dimension == 0) || */(dimension == 1));
    for (y = 0; y < a->shape->sizes[0]; y++)
    {
        ((float *)self->data)[y] = 0.0f;
        for (x = 0; x < a->shape->sizes[1]; x++)
        {
            ((float *)self->data)[y] += ((float *)a->data)[y * a->shape->sizes[1] + x];
        }
    }

    return self;
}

LeTensor *
le_matrix_new_one_hot(LeTensor *a, unsigned num_classes)
{
    assert(a->shape->num_dimensions == 2);
    assert(a->shape->sizes[0] == 1);
    
    unsigned example, klass;
    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = le_shape_new(2, num_classes, a->shape->sizes[1]);
    self->stride = le_shape_get_last_size(self->shape);
    self->owns_data = true;
    self->data = malloc(le_shape_get_elements_count(self->shape) * le_type_size(self->element_type));
    
    for (example = 0; example < a->shape->sizes[1]; example++)
    {
        for (klass = 0; klass < num_classes; klass++)
        {
            float label = (klass == le_tensor_u8_at(a, example)) ? 1.0f : 0.0f;
            le_matrix_set_element(self, klass, example, label);
        }
    }

    return self;
}

LeTensor *
le_matrix_new_product(LeTensor *a, LeTensor *b)
{
    return le_matrix_new_product_full(a, false, b, false);
}

LeTensor *
le_matrix_new_product_full(LeTensor *a, bool transpose_a, LeTensor *b, bool transpose_b)
{
#ifdef __APPLE__
    return le_accelerate_matrix_new_product(a, transpose_a, b, transpose_b);
#else
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);

    unsigned x;
    unsigned y;
    unsigned i;
    
    LeTensor *self;
    
    assert(a->element_type == b->element_type);
    
    unsigned a_width = transpose_a ? a->shape->sizes[0] : a->shape->sizes[1];
    unsigned a_height = transpose_a ? a->shape->sizes[1] : a->shape->sizes[0];
    unsigned b_width = transpose_b ? b->shape->sizes[0] : b->shape->sizes[1];
    unsigned b_height = transpose_b ? b->shape->sizes[1] : b->shape->sizes[0];
    
    assert(a_width == b_height);
            
    self = malloc(sizeof(struct LeTensor));
    self->element_type = a->element_type;
    self->shape = le_shape_new(2, a_height, b_width);
    self->stride = le_shape_get_last_size(self->shape);
    self->owns_data = true;
    self->data = malloc(le_shape_get_elements_count(self->shape) * sizeof(float));
    
    for (y = 0; y < a_height; y++)
    {
        for (x = 0; x < b_width; x++)
        {
            size_t index = y * b_width + x;
            ((float *)self->data)[index] = 0.0f;
            for (i = 0; i < a_width; i++)
            {
                /// @note: Check indices
                size_t a_index = transpose_a ? i * a_height + y : y * a_width + i;
                float a_element = ((float *)a->data)[a_index];
                size_t b_index = transpose_b ? x * b_height + i : i * b_width + x;
                float b_element = ((float *)b->data)[b_index];
                float prod = a_element * b_element;
                ((float *)self->data)[index] += prod;
            }
        }
    }
    
    return self;
#endif
}

LeTensor *
le_matrix_get_column(LeTensor *matrix, unsigned x)
{
    assert(matrix->shape->num_dimensions == 2);
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = matrix->element_type;
    self->shape = le_shape_new(2, matrix->shape->sizes[0], 1);
    self->stride = matrix->stride;

    self->owns_data = false;
    self->data = matrix->data + x * le_type_size(self->element_type);
    
    return self;
}
                  
LeTensor *
le_matrix_get_column_copy(LeTensor *self, unsigned x)
{
    assert(self->shape->num_dimensions == 2);
    /// @todo: Add dimension checks
    
    unsigned y;
    unsigned height = le_matrix_get_height(self);
    LeTensor *column = le_matrix_new_uninitialized(height, 1);
    
    for (y = 0; y < height; y++)
    {
        ((float *)column->data)[y] = ((float *)self->data)[y * self->shape->sizes[1] + x];
    }
    
    return column;
}
