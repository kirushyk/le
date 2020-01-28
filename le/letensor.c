/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "letensor.h"
#include "letensor-imp.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#ifdef __APPLE__
#include "../platform/accelerate/leaccelerate.h"
#endif

LeTensor *
le_tensor_new(void)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_VOID;
    self->shape = NULL;
    self->stride = 0;
    self->owns_data = false;
    self->data = NULL;
    return self;
}

LeTensor *
le_scalar_new_f32(float scalar)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = le_shape_new(0);
    self->stride = 0;
    self->owns_data = true;
    self->data = malloc(sizeof(float));
    *((float *)self->data) = scalar;
    return self;
}

LeTensor *
le_tensor_new_from_data(LeType element_type, LeShape *shape, void *data)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = element_type;
    self->shape = shape;
    self->stride = le_shape_get_last_size(self->shape);
    self->owns_data = true;
    self->data = data;
    return self;
}

LeTensor *
le_tensor_new_copy(LeTensor *another)
{
    assert(another);
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = another->element_type;
    self->shape = le_shape_copy(another->shape);
    self->stride = another->stride;
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
    self->stride = another->stride;
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
    self->stride = another->stride;
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
le_tensor_equal(LeTensor *a, LeTensor *b)
{
    /// @todo: Take stride into account
    if (a == b)
        return true;
    
    if ((a && !b) || (!a && b))
        return false;
    
    if (a->element_type != b->element_type)
        return false;
    
    /// @note: Stride may not be equal
    
    if (!le_shape_equal(a->shape, b->shape))
        return false;
    
    if (memcmp(a->data, b->data, le_shape_get_elements_count(a->shape) * le_type_size(a->element_type)))
        return false;
    
    return true;
}

bool     
le_tensor_reshape(LeTensor *self, unsigned num_dimensions, ...)
{
    /// @todo: Take stride into account
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
    /// @todo: Take stride into account
    if (!another)
        return NULL;
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = another->element_type;
    self->shape = le_shape_lower_dimension(another->shape);
    self->stride = le_shape_get_last_size(self->shape);

    size_t data_size = le_shape_get_elements_count(self->shape) * le_type_size(self->element_type);
    self->owns_data = false;
    self->data = another->data + index * data_size;
    
    return self;
}

LeTensor *
le_tensor_pick_copy(LeTensor *another, uint32_t index)
{
    /// @todo: Take stride into account
    if (!another)
        return NULL;
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = another->element_type;
    self->shape = le_shape_lower_dimension(another->shape);
    self->stride = le_shape_get_last_size(self->shape);
    
    size_t data_size = le_shape_get_elements_count(self->shape) * le_type_size(self->element_type);
    self->owns_data = true;
    self->data = malloc(data_size);
    
    memcpy(self->data, another->data + index * data_size, data_size);
    
    return self;
}

uint8_t
le_tensor_u8_at(LeTensor *tensor, uint32_t index)
{
    /// @todo: Take stride into account
    assert(tensor->element_type == LE_TYPE_UINT8);

    return ((uint8_t *)tensor->data)[index];
}

float
le_tensor_f32_at(LeTensor *tensor, uint32_t index)
{
    /// @todo: Take stride into account
    assert(tensor->element_type == LE_TYPE_FLOAT32);
    
    return ((float *)tensor->data)[index];
}

void
le_tensor_f32_set(LeTensor *tensor, uint32_t index, float value)
{
    /// @todo: Take stride into account
    assert(tensor->element_type == LE_TYPE_FLOAT32);
    
    ((float *)tensor->data)[index] = value;
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
#ifdef __APPLE__
    return le_accelerate_dot_product(a, b);
#else
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    
    unsigned y;
    
    float result = 0;

    /** @todo: Test results against transposed a multiplied by b */
    assert(a->shape->sizes[0] == b->shape->sizes[0]);
    assert(a->shape->sizes[1] == 1);
    assert(b->shape->sizes[1] == 1);
    
    for (y = 0; y < a->shape->sizes[0]; y++)
    {
        /** @note: This addressing is correct as we
            ensured that widths of both matrices
            (supposed to be column vectors) is 1 */
        // result += ((float *)a->data)[y] * ((float *)b->data)[y];
        /** @note: Stride (separate from width) added */
        result += ((float *)a->data)[y * a->stride] * ((float *)b->data)[y * b->stride];
    }
    
    return result;
#endif
}

float
le_rbf(LeTensor *a, LeTensor *b, float sigma)
{
#ifdef __APPLE__
    return le_accelerate_rbf(a, b, sigma);
#else
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);

    float result = 0;
    
    /** @todo: Test results against transposed a multiplied by b */
    assert(a->shape->sizes[0] == b->shape->sizes[0]);
    assert(a->shape->sizes[1] == 1);
    assert(b->shape->sizes[1] == 1);
    
    for (unsigned y = 0; y < a->shape->sizes[0]; y++)
    {
        float sub = ((float *)a->data)[y * a->stride] - ((float *)b->data)[y * b->stride];
        result += sub * sub;
    }
    
    return expf(-result / (2.0f * sigma * sigma));
#endif
}

void
le_tensor_add(LeTensor *a, LeTensor *b)
{
    /// @todo: Take stride into account
    assert(le_shape_equal(a->shape, b->shape));
    
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(a->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)a->data)[i] += ((float *)b->data)[i];
    }
}

void
le_tensor_subtract(LeTensor *a, LeTensor *b)
{
    /// @todo: Take stride into account
    assert(le_shape_equal(a->shape, b->shape));
    
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(a->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)a->data)[i] -= ((float *)b->data)[i];
    }
}

void
le_tensor_subtract_scaled(LeTensor *a, float scale, LeTensor *b)
{
    /// @todo: Take stride into account
    assert(le_shape_equal(a->shape, b->shape));
    
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(a->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)a->data)[i] -= scale * ((float *)b->data)[i];
    }
}

void
le_tensor_multiply_by_scalar(LeTensor *self, float b)
{
    /// @todo: Take stride into account
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
    /// @todo: Take stride into account
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] += b;
    }
}

void
le_tensor_subtract_scalar(LeTensor *self, float b)
{
    /// @todo: Take stride into account
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] -= b;
    }
}

float
le_tensor_sum(LeTensor *self)
{
    /// @todo: Take stride into account
    float sum = 0.0;
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        sum += ((float *)self->data)[i];
    }
    
    return sum;
}

#ifndef NOT__APPLE__
static float
le_sigmoid(const float a)
{
    return 1.0 / (1.0 + exp(-a));
}
#endif

void
le_tensor_apply_sigmoid(LeTensor *self)
{
    /// @todo: Take stride into account
#ifdef __APPLE__
    return le_accelerate_tensor_apply_sigmoid(self);
#else
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = le_sigmoid(((float *)self->data)[i]);
    }
#endif
}

void
le_tensor_apply_sigmoid_prime(LeTensor *self)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        float sigmoid = le_sigmoid(((float *)self->data)[i]);
        ((float *)self->data)[i] = sigmoid * (1.0f - sigmoid);
    }
}

void
le_tensor_apply_tanh(LeTensor *self)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = tanhf(((float *)self->data)[i]);
    }
}

void
le_tensor_apply_sqr(LeTensor *self)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = ((float *)self->data)[i] * ((float *)self->data)[i];
    }
}

void
le_tensor_apply_1_minus(LeTensor *self)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = 1.0f - ((float *)self->data)[i];
    }
}

void
le_tensor_apply_greater_than(LeTensor *self, float scalar)
{
    /// @todo: Take stride into account
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
    /// @todo: Take stride into account
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = ((float *)self->data)[i] > 0.0f ? 1.0f : -1.0f;
    }
}

const char *
le_tensor_to_cstr(LeTensor *self)
{
    /// @todo: Fix buffer overflow
    static char buffer[1024];

    if (self->shape->num_dimensions != 2)
    {
        sprintf(buffer, "<%dD tensor>\n", self->shape->num_dimensions);
        return buffer;
    }
    
    unsigned x;
    unsigned y;

    char *ptr = buffer;
    ptr[0] = '[';
    ptr++;

    for (y = 0; (y < self->shape->sizes[0]) && (y <= 5); y++)
    {
        for (x = 0; (x < self->shape->sizes[1]) && (x <= 5); x++)
        {
            int written = 0;
            sprintf(ptr, "%1.3f%n", ((float *)self->data)[y * self->shape->sizes[1] + x], &written);
            ptr += written;
            if (x < self->shape->sizes[1] - 1)
            {
                *ptr = ' ';
                ptr++;
            }
        }
        if (x < self->shape->sizes[1])
        {
            int written = 0;
            sprintf(ptr, "...%n", &written);
            ptr += written;
        }
        if (y < self->shape->sizes[0] - 1)
        {
            int written = 0;
            sprintf(ptr, ";\n %n", &written);
            ptr += written;
        }
    }
    if (y < self->shape->sizes[0])
    {
        int written = 0;
        sprintf(ptr, " ...\n%n", &written);
        ptr += written;
    }
    sprintf(ptr, "]");

    return buffer;
}

/** @note: Temporary */
void
le_tensor_print(LeTensor *self, FILE *stream)
{
    /// @todo: Take stride into account
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
