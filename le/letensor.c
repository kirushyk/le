/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "letensor.h"
#include "letensor-imp.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

LeTensor *
le_tensor_new(void)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_VOID;
    self->shape = NULL;
    self->owns_data = false;
    self->data = NULL;
    return self;
}

LeTensor *
le_tensor_new_from_data(LeType element_type, LeShape *shape, void *data)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = element_type;
    self->shape = shape;
    self->owns_data = true;
    self->data = data;
    return self;
}

LeTensor *
le_tensor_new_copy(LeTensor *another)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = another->element_type;
    self->shape = le_shape_copy(another->shape);
    self->owns_data = true;
    size_t data_size = le_shape_get_elements_count(another->shape) * le_type_size(another->element_type);
    self->data = malloc(data_size);
    memcpy(self->data, another->data, data_size);
    return self;
}

LeTensor *
le_tensor_new_cast_f32(LeTensor *another)
{
    unsigned i;
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = le_shape_copy(another->shape);
    self->owns_data = true;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    size_t data_size = elements_count * le_type_size(self->element_type);
    self->data = malloc(data_size);
    
    /// @todo: Add support for types other than UINT8
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = ((uint8_t *)another->data)[i] * (2.0f / 255.0f) - 1.0f;
    }
    
    return self;
}

LeTensor *
le_tensor_new_f32_equal_u8(LeTensor *another, uint8_t scalar)
{
    unsigned i;
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = le_shape_copy(another->shape);
    self->owns_data = true;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    size_t data_size = elements_count * le_type_size(self->element_type);
    self->data = malloc(data_size);
    
    /// @todo: Add support for types other than UINT8
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = (((uint8_t *)another->data)[i] == scalar) ? 1.0f : 0.0f;
    }
    
    return self;
}

bool     
le_tensor_reshape(LeTensor *self, unsigned num_dimensions, ...)
{
    /// @todo: Add more assertions

    uint32_t *sizes = malloc(num_dimensions * sizeof(uint32_t));
    va_list args;
    va_start(args, num_dimensions);
    
    for (uint8_t i = 0; i < num_dimensions; i++)
    {
        int size = va_arg(args, int);
        sizes[i] = size;
    }
    
    va_end(args);
    
    LeShape *new_shape = le_shape_new_from_data(num_dimensions, sizes);

    if (le_shape_get_elements_count(new_shape) == le_shape_get_elements_count(self->shape))
    {
        le_shape_free(self->shape);
        self->shape = new_shape;

        return true;
    }
    else
    {
        le_shape_free(self->shape);

        return false;
    }
}

LeTensor *
le_tensor_pick(LeTensor *another, uint32_t index)
{
    if (!another)
        return NULL;
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = another->element_type;
    self->shape = le_shape_lower_dimension(another->shape);

    size_t data_size = le_shape_get_elements_count(self->shape) * le_type_size(self->element_type);
    self->owns_data = false;
    self->data = another->data + index * data_size;
    
    return self;
}

LeTensor *
le_tensor_pick_copy(LeTensor *another, uint32_t index)
{
    if (!another)
        return NULL;
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = another->element_type;
    self->shape = le_shape_lower_dimension(another->shape);
    
    size_t data_size = le_shape_get_elements_count(self->shape) * le_type_size(self->element_type);
    self->owns_data = true;
    self->data = malloc(data_size);
    
    memcpy(self->data, another->data + index * data_size, data_size);
    
    return self;
}

uint8_t
le_tensor_at(LeTensor *tensor, uint32_t index)
{
    return ((uint8_t *)tensor->data)[index];
}

void
le_matrix_empty(LeTensor *self)
{
    free(self->data);
    self->data = NULL;
    free(self->shape);
    self->shape = NULL;
    self->element_type = LE_TYPE_VOID;
}

float
le_dot_product(LeTensor *a, LeTensor *b)
{
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    
    unsigned y;
    
    float result = 0;

    /** @todo: Test results against transposed a multiplied by b */
    if (a->shape->sizes[0] != b->shape->sizes[0] || a->shape->sizes[1] != 1 || b->shape->sizes[1] != 1)
        return nanf("");
    
    for (y = 0; y < a->shape->sizes[0]; y++)
    {
        /** @note: This addressing is correct as we
            ensured that widths of both matrices
            (supposed to be column vectors) is 1 */
        result += ((float *)a->data)[y] * ((float *)b->data)[y];
    }
    
    return result;
}

float
le_rbf(LeTensor *a, LeTensor *b, float sigma)
{
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);

    float result = 0;
    
    /** @todo: Test results against transposed a multiplied by b */
    if (a->shape->sizes[0] != b->shape->sizes[0] || a->shape->sizes[1] != 1 || b->shape->sizes[1] != 1)
        return nanf("");
    
    for (unsigned y = 0; y < a->shape->sizes[0]; y++)
    {
        float sub = ((float *)a->data)[y] - ((float *)b->data)[y];
        result += sub * sub;
    }
    
    return expf(-result / (2.0f * sigma * sigma));
}

void
le_tensor_subtract(LeTensor *a, LeTensor *b)
{
    if (le_shape_equal(a->shape, b->shape))
    {
        unsigned i;
        unsigned elements_count = le_shape_get_elements_count(a->shape);
        
        for (i = 0; i < elements_count; i++)
        {
            ((float *)a->data)[i] -= ((float *)b->data)[i];
        }
    }
}

void
le_tensor_multiply_by_scalar(LeTensor *self, float b)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] *= b;
    }
}

void
le_tensor_add_scalar(LeTensor *self, float b)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] += b;
    }
}

float
le_tensor_sum(LeTensor *self)
{
    float sum = 0.0;
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        sum += ((float *)self->data)[i];
    }
    
    return sum;
}

static float
le_sigmoid(const float a)
{
    return 1.0 / (1.0 + exp(-a));
}

void
le_tensor_apply_sigmoid(LeTensor *self)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = le_sigmoid(((float *)self->data)[i]);
    }
}

void
le_tensor_apply_greater_than(LeTensor *self, float scalar)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = ((float *)self->data)[i] > scalar ? 1.0f : 0.0f;
    }
}

void
le_tensor_apply_svm_prediction(LeTensor *self)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = ((float *)self->data)[i] > 0.0f ? 1.0f : -1.0f;
    }
}

/** @note: Temporary */
void
le_matrix_print(LeTensor *self, FILE *stream)
{
    if (self->shape->num_dimensions != 2)
    {
        fprintf(stream, "<%dD tensor>\n", self->shape->num_dimensions);
        return;
    }
    
    unsigned x;
    unsigned y;
    fprintf(stream, "[");
    for (y = 0; y < self->shape->sizes[0]; y++)
    {
        for (x = 0; x < self->shape->sizes[1]; x++)
        {
            fprintf(stream, "%1.3f", ((float *)self->data)[y * self->shape->sizes[1] + x]);
            if (x < self->shape->sizes[1] - 1)
            {
                fprintf(stream, " ");
            }
        }
        if (y < self->shape->sizes[0] - 1)
        {
            fprintf(stream, ";\n ");
        }
    }
    fprintf(stream, "]\n");
}

void
le_tensor_free(LeTensor *self)
{
    if (self->owns_data)
    {
        free(self->data);
    }
    free(self->shape);
    free(self);
}
