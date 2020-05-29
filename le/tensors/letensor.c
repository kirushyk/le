/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "tensor"

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
le_tensor_new_from_va_list(LeType element_type, unsigned num_dimensions, va_list dims_and_data)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = element_type;
        
    uint32_t *sizes = malloc(num_dimensions * sizeof(uint32_t));
    for (unsigned i = 0; i < num_dimensions; i++)
    {
        int size = va_arg(dims_and_data, int);
        sizes[i] = size;
    }
    self->shape = le_shape_new_from_data(num_dimensions, sizes);
    self->stride = le_shape_get_last_size(self->shape);
    
    self->owns_data = true;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    self->data = malloc(elements_count * le_type_size(self->element_type));

    if (self->element_type == LE_TYPE_FLOAT16)
        LE_ERROR("F16 Tensor init from va_list not implemented");
        
    for (unsigned i = 0; i < elements_count; i++)
    {
        switch (self->element_type)
        {
        case LE_TYPE_INT8:
            {
                int8_t value = (int8_t)va_arg(dims_and_data, int);
                ((int8_t *)self->data)[i] = value;
            }
            break;
        case LE_TYPE_UINT8:
            {
                uint8_t value = (uint8_t)va_arg(dims_and_data, int);
                ((uint8_t *)self->data)[i] = value;
            }
            break;
        case LE_TYPE_INT16:
            {
                int16_t value = (int16_t)va_arg(dims_and_data, int);
                ((int16_t *)self->data)[i] = value;
            }
            break;
        case LE_TYPE_UINT16:
            {
                uint16_t value = (uint16_t)va_arg(dims_and_data, int);
                ((uint16_t *)self->data)[i] = value;
            }
            break;
        case LE_TYPE_INT32:
            {
                int32_t value = (int32_t)va_arg(dims_and_data, int);
                ((int32_t *)self->data)[i] = value;
            }
            break;
        case LE_TYPE_UINT32:
            {
                int32_t value = (int32_t)va_arg(dims_and_data, int);
                ((int32_t *)self->data)[i] = value;
            }
            break;
        case LE_TYPE_FLOAT32:
            {
                float value = (float)va_arg(dims_and_data, double);
                ((float *)self->data)[i] = value;
            }
            break;
        case LE_TYPE_FLOAT64:
            {
                double value = (double)va_arg(dims_and_data, double);
                ((double *)self->data)[i] = value;
            }
            break;
        default:
            break;
        }
    }

    return self;
}

LeTensor *
le_tensor_new(LeType element_type, unsigned num_dimensions, ...)
{
    LeTensor *self;
        
    va_list args;
    va_start(args, num_dimensions);
    self = le_tensor_new_from_va_list(element_type, num_dimensions, args);
    va_end(args);

    return self;
}

LeTensor *
le_tensor_new_rand_f32(LeShape *shape)
{
    unsigned i;
    unsigned elements_count;
    LeTensor *self;
    
    self = malloc(sizeof(struct LeTensor));
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = shape;
    self->stride = le_shape_get_last_size(self->shape);
    self->owns_data = true;
    elements_count = le_shape_get_elements_count(shape);
    self->data = malloc(elements_count * sizeof(float));
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = rand() / (float)RAND_MAX;
    }
    
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
le_tensor_new_copy(const LeTensor *another)
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
le_tensor_new_equal_u8(LeType type, LeTensor *another, uint8_t scalar)
{
    unsigned i;
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = type;
    self->shape = le_shape_copy(another->shape);
    self->stride = another->stride;
    self->owns_data = true;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    size_t data_size = elements_count * le_type_size(self->element_type);
    self->data = malloc(data_size);
    
    /// @todo: Add support for types other than UINT8
    for (i = 0; i < elements_count; i++)
    {
        bool equal = (((uint8_t *)another->data)[i] == scalar);
        ((float *)self->data)[i] = equal ? 1.0f : 0.0f;
    }
    
    return self;
}

bool
le_tensor_equal(const LeTensor *a, const LeTensor *b)
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
le_tensor_pick_copy(const LeTensor *another, uint32_t index)
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
le_tensor_at_u8(const LeTensor *tensor, uint32_t index)
{
    /// @todo: Take stride into account
    assert(tensor->element_type == LE_TYPE_UINT8);

    return ((uint8_t *)tensor->data)[index];
}

float
le_tensor_at_f32(const LeTensor *tensor, uint32_t index)
{
    /// @todo: Take stride into account
    assert(tensor->element_type == LE_TYPE_FLOAT32);
    
    return ((float *)tensor->data)[index];
}

void
le_tensor_set_f32(LeTensor *tensor, uint32_t index, float value)
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
    le_shape_free(self->shape);
    self->shape = NULL;
    self->element_type = LE_TYPE_VOID;
}

float
le_dot_product(const LeTensor *a, const LeTensor *b)
{
#ifdef __APPLE__
    return le_accelerate_dot_product(a, b);
#else
    assert(a->element_type == LE_TYPE_FLOAT32);
    assert(b->element_type == LE_TYPE_FLOAT32);
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
le_rbf(const LeTensor *a, const LeTensor *b, float sigma)
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
le_tensor_add_tensor(LeTensor *a, LeTensor *b)
{
    /// @todo: Take stride into account
    assert(a->element_type == LE_TYPE_FLOAT32);
    assert(b->element_type == LE_TYPE_FLOAT32);
    assert(le_shape_equal(a->shape, b->shape));
    
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(a->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)a->data)[i] += ((float *)b->data)[i];
    }
}

void
le_tensor_sub_f32(LeTensor *self, float b)
{
    assert(self->element_type == LE_TYPE_FLOAT32);

    /// @todo: Take stride into account
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] -= b;
    }
}

void
le_tensor_sub_tensor(LeTensor *a, const LeTensor *b)
{
    /// @todo: Take stride into account
    assert(a->element_type == LE_TYPE_FLOAT32);
    assert(b->element_type == LE_TYPE_FLOAT32);
    assert(le_shape_equal(a->shape, b->shape));
    
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(a->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)a->data)[i] -= ((float *)b->data)[i];
    }
}

void
le_tensor_sub_scaled_f32(LeTensor *a, float scale, const LeTensor *b)
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
le_tensor_mul_f32(LeTensor *self, float b)
{
    assert(self->element_type == LE_TYPE_FLOAT32);

    /// @todo: Take stride into account
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] *= b;
    }
}

void        
le_tensor_mul_tensor(LeTensor *self, const LeTensor *b)
{
    assert(self->element_type == LE_TYPE_FLOAT32);
    assert(b->element_type == LE_TYPE_FLOAT32);
    assert(le_shape_equal(self->shape, b->shape));

    /// @todo: Take stride into account
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] *= ((float *)b->data)[i];
    }
}

void
le_tensor_add_f32(LeTensor *self, float b)
{
    assert(self->element_type == LE_TYPE_FLOAT32);
    /// @todo: Take stride into account
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] += b;
    }
}

float
le_tensor_sum_f32(const LeTensor *self)
{
    assert(self->element_type == LE_TYPE_FLOAT32);
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

#ifndef __APPLE__
static float
le_sigmoid(const float a)
{
    return 1.0 / (1.0 + expf(-a));
}
#endif

void
le_tensor_apply_sigmoid(LeTensor *self)
{
    /// @todo: Take stride into account
#ifdef __APPLE__
    return le_accelerate_tensor_apply_sigmoid(self);
#else
    assert(self->element_type == LE_TYPE_FLOAT32);
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
    /// @todo: Take stride into account
#ifdef __APPLE__
    return le_accelerate_tensor_apply_sigmoid_prime(self);
#else
    assert(self->element_type == LE_TYPE_FLOAT32);
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        float sigmoid = le_sigmoid(((float *)self->data)[i]);
        ((float *)self->data)[i] = sigmoid * (1.0f - sigmoid);
    }
#endif
}

void
le_tensor_apply_tanh(LeTensor *self)
{
    assert(self->element_type == LE_TYPE_FLOAT32 ||
           self->element_type == LE_TYPE_FLOAT64);
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        switch (self->element_type)
        {
        case LE_TYPE_FLOAT32:
            ((float *)self->data)[i] = tanhf(((float *)self->data)[i]);
            break;
        case LE_TYPE_FLOAT64:
            ((double *)self->data)[i] = tanh(((double *)self->data)[i]);
            break;
        default:
            return;
        }
    }
}

void
le_tensor_apply_sqr(LeTensor *self)
{
    assert(self->element_type == LE_TYPE_FLOAT32 ||
           self->element_type == LE_TYPE_FLOAT64);

    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        switch (self->element_type)
        {
        case LE_TYPE_FLOAT32:
            ((float *)self->data)[i] = ((float *)self->data)[i] * ((float *)self->data)[i];
            break;
        case LE_TYPE_FLOAT64:
            ((double *)self->data)[i] = ((double *)self->data)[i] * ((double *)self->data)[i];
            break;
        default:
            return;
        }
    }
}

void
le_tensor_apply_1_minus(LeTensor *self)
{
    assert(self->element_type == LE_TYPE_FLOAT32 ||
           self->element_type == LE_TYPE_FLOAT64);

    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        switch (self->element_type)
        {
        case LE_TYPE_FLOAT32:
            ((float *)self->data)[i] = 1.0f - ((float *)self->data)[i];
            break;
        case LE_TYPE_FLOAT64:
            ((double *)self->data)[i] = 1.0f - ((double *)self->data)[i];
            break;
        default:
            return;
        }
    }
}

void
le_tensor_apply_x_minus_sqr_x(LeTensor *self)
{
    assert(self->element_type == LE_TYPE_FLOAT32 ||
           self->element_type == LE_TYPE_FLOAT64);

    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        switch (self->element_type)
        {
        case LE_TYPE_FLOAT32:
            {
                float x = ((float *)self->data)[i];
                ((float *)self->data)[i] = x * (1 - x);
            }
            break;
        case LE_TYPE_FLOAT64:
            {
                double x = ((double *)self->data)[i];
                ((double *)self->data)[i] = x * (1 - x);
            }
            break;
        default:
            return;
        }
    }
}

void
le_tensor_apply_gt_f32(LeTensor *self, float scalar)
{
    assert(self->element_type == LE_TYPE_FLOAT32 ||
           self->element_type == LE_TYPE_FLOAT64);
    
    /// @todo: Take stride into account
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        switch (self->element_type)
        {
        case LE_TYPE_FLOAT32:
            ((float *)self->data)[i] = ((float *)self->data)[i] > scalar ? 1.0f : 0.0f;
            break;
        case LE_TYPE_FLOAT64:
            ((double *)self->data)[i] = ((double *)self->data)[i] > scalar ? 1.0 : 0.0;
            break;
        default:
            return;
        }
    }
}

void
le_tensor_apply_sgn(LeTensor *self)
{
    assert(self->element_type == LE_TYPE_FLOAT32);

    /// @todo: Take stride into account
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((float *)self->data)[i] = ((float *)self->data)[i] > 0.0f ? 1.0f : -1.0f;
    }
}

void
le_tensor_apply_relu(LeTensor *self)
{
    /// @note: Not implemented for half precision floating point values (float16)
    /// @note: There is no sense in applying ReLU to Tensors of unsigned integers.
    assert(self->element_type == LE_TYPE_FLOAT32 ||
           self->element_type == LE_TYPE_FLOAT64 ||
           self->element_type == LE_TYPE_INT8 ||
           self->element_type == LE_TYPE_INT16 ||
           self->element_type == LE_TYPE_INT32);

    /// @todo: Take stride into account
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        switch (self->element_type)
        {
#define APPLY_RELU(T) { T value = ((T *)self->data)[i]; ((T *)self->data)[i] = value > 0 ? value : 0;  }
        case LE_TYPE_FLOAT32:
            APPLY_RELU(float)
            break;
        case LE_TYPE_FLOAT64:
            APPLY_RELU(double)
            break;
        case LE_TYPE_INT8:
            APPLY_RELU(int8_t)
            break;
        case LE_TYPE_INT16:
            APPLY_RELU(int16_t)
            break;
        case LE_TYPE_INT32:
            APPLY_RELU(int32_t)
            break;
        default:
            return;
#undef APPLY_RELU
        }
    }
}

/// @section ugly

#define TENSOR_PRINT_MAX_SIZE 10
#define BUFFER_SIZE 1024

const char *
le_tensor_to_cstr(const LeTensor *self)
{
    /// @todo: Fix buffer overflow
    static char buffer[BUFFER_SIZE];

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

    for (y = 0; (y < self->shape->sizes[0]) && (y < TENSOR_PRINT_MAX_SIZE); y++)
    {
        for (x = 0; (x < self->shape->sizes[1]) && (x < TENSOR_PRINT_MAX_SIZE); x++)
        {
            if (ptr > (buffer + BUFFER_SIZE - 256))
                goto too_long;

            int written = 0;
            switch (self->element_type)
            {
                case LE_TYPE_UINT8:
                    sprintf(ptr, "%u%n", (unsigned)((uint8_t *)self->data)[y * self->shape->sizes[1] + x], &written);
                    break;
                case LE_TYPE_INT8:
                    sprintf(ptr, "%d%n", (int)((int8_t *)self->data)[y * self->shape->sizes[1] + x], &written);
                    break;
                case LE_TYPE_INT16:
                    sprintf(ptr, "%d%n", (int)((int16_t *)self->data)[y * self->shape->sizes[1] + x], &written);
                    break;
                case LE_TYPE_INT32:
                    sprintf(ptr, "%d%n", (int)((int32_t *)self->data)[y * self->shape->sizes[1] + x], &written);
                    break;
                case LE_TYPE_FLOAT32:
                    sprintf(ptr, "%1.3f%n", ((float *)self->data)[y * self->shape->sizes[1] + x], &written);
                    break;
                case LE_TYPE_FLOAT64:
                    sprintf(ptr, "%1.3lf%n", ((double *)self->data)[y * self->shape->sizes[1] + x], &written);
                    break;
                case LE_TYPE_VOID:
                default:
                    sprintf(ptr, "?%n", &written);
                    break;
            }
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
    
too_long:
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
le_tensor_print(const LeTensor *self, FILE *stream)
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
    if (self == NULL)
        return;
        
    if (self->owns_data)
        free(self->data);
    
    le_shape_free(self->shape);
    free(self);
}

LeTensorStats
le_tensor_get_stats(LeTensor *self)
{
    LeTensorStats stats;
    stats.deviation = 0.0f;
    stats.mean = 0.0f;
    stats.max = 0.0f;
    stats.min = 0.0f;

    /// @todo: Take stride into account
    unsigned elements_count = le_shape_get_elements_count(self->shape);

    if (elements_count >= 1)
    {
        float value = ((float *)self->data)[0];
        stats.max = value;
        stats.min = value;
        stats.mean = value;
        for (unsigned i = 1; i < elements_count; i++)
        {
            float value = ((float *)self->data)[i];
            if (value > stats.max)
                stats.max = value;
            if (value < stats.min)
                stats.min = value; 
            stats.mean += value;
        }
        stats.mean /= elements_count;
        for (unsigned i = 1; i < elements_count; i++)
        {
            float value = ((float *)self->data)[i];
            stats.deviation += fabs(value - stats.mean);
        }
        stats.deviation /= elements_count;
    }

    return stats;
}