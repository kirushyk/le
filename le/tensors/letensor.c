/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "tensor"

#include "../config.h"
#include <le/lelog.h>
#include "letensor.h"
#include "letensor-imp.h"
#include "letensor-cast.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#ifdef __APPLE__
#   include "../backends/accelerate/leaccelerate.h"
#elif defined(HAVE_OPENBLAS)
#   include "../backends/openblas/leopenblas.h"
#endif
#if defined(HAVE_METAL)
#   include "../backends/metal/lemetal.h"
#endif
#if defined(HAVE_CUDA)
#   include "../backends/cuda/lecuda.h"
#endif

LeTensor *
le_tensor_new_from_va_list(LeType element_type, unsigned num_dimensions, va_list dims_and_data)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->device_type = LE_DEVICE_TYPE_CPU;
    self->element_type = element_type;
        
    self->shape = le_shape_new_uninitialized(num_dimensions);
    for (unsigned i = 0; i < num_dimensions; i++)
    {
        int size = va_arg(dims_and_data, int);
        le_shape_set_size(self->shape, i, size);
    }
    self->stride = le_shape_get_size(self->shape, -1);
    
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
    self->device_type = LE_DEVICE_TYPE_CPU;
    self->element_type = LE_TYPE_FLOAT32;
    self->shape = shape;
    self->stride = le_shape_get_size(self->shape, -1);
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
le_tensor_new_uninitialized(LeType element_type, LeShape *shape)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->device_type = LE_DEVICE_TYPE_CPU;
    self->element_type = element_type;
    self->shape = shape;
    self->stride = le_shape_get_size(self->shape, -1);
    self->owns_data = true;
    size_t element_size = le_type_size(element_type);
    unsigned elements_count = le_shape_get_elements_count(shape);
    self->data = malloc(element_size * elements_count);
    return self;
}

void *
le_tensor_get_data(const LeTensor *self)
{
    return self->data;
}

LeTensor *
le_tensor_new_copy(const LeTensor *another)
{
    assert(another);
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->device_type = another->device_type;
    self->element_type = another->element_type;
    self->shape = le_shape_copy(another->shape);
    self->stride = le_shape_get_size(self->shape, -1);
    self->owns_data = true;
    size_t data_size = le_shape_get_elements_count(self->shape) * le_type_size(self->element_type);
    switch (self->device_type)
    {
#ifdef HAVE_METAL
    case LE_DEVICE_TYPE_METAL:
        if (another->stride == le_shape_get_size(another->shape, -1))
        {
            self->data = le_metal_data_copy(another->data, data_size);
        }
        break;
#endif
#ifdef HAVE_CUDA
    case LE_DEVICE_TYPE_CUDA:
        if (another->stride == le_shape_get_size(another->shape, -1))
        {
            self->data = le_cuda_data_copy(another->data, data_size);
        }
        break;
#endif
    case LE_DEVICE_TYPE_CPU:
        {
            self->data = malloc(data_size);
            if (another->stride == le_shape_get_size(another->shape, -1))
            {
                memcpy(self->data, another->data, data_size);
            }
            else
            {
                unsigned regions_count = le_shape_get_regions_count(another->shape);
                size_t region_size = self->stride * le_type_size(self->element_type);
                size_t bytes_stride = another->stride * le_type_size(another->element_type);
                for (unsigned i = 0; i < regions_count; i++)
                {
                    memcpy((uint8_t *)self->data + i * region_size,
                        (uint8_t *)another->data + i * bytes_stride,
                        region_size);
                }
            }
        }
        break;
    default:
        assert(false);
        break;
    }
    
    return self;
}

LeTensor *
le_tensor_new_zeros(LeType element_type, LeShape *shape)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->device_type = LE_DEVICE_TYPE_CPU;
    self->element_type = element_type;
    self->shape = shape;
    self->stride = le_shape_get_size(self->shape, -1);
    self->owns_data = true;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    size_t data_size = elements_count * le_type_size(self->element_type);
    self->data = malloc(data_size);
    for (unsigned i = 0; i < elements_count; i++)
    {
        switch (self->element_type)
        {
        case LE_TYPE_INT8:
        case LE_TYPE_UINT8:
            ((int8_t *)self->data)[i] = 0;
            break;
        case LE_TYPE_INT16:
        case LE_TYPE_UINT16:
            ((int16_t *)self->data)[i] = 0;
            break;
        case LE_TYPE_INT32:
        case LE_TYPE_UINT32:
            ((int32_t *)self->data)[i] = 0;
            break;
        case LE_TYPE_FLOAT16:
            ((uint16_t *)self->data)[i] = F16_0;
            break;
        case LE_TYPE_FLOAT32:
            ((float *)self->data)[i] = 0.0f;
            break;
        case LE_TYPE_FLOAT64:
            ((double *)self->data)[i] = 0.0;
            break;
        default:
            break;
        }
    }
    return self;
}

LeTensor *
le_tensor_new_zeros_like(const LeTensor *another)
{
    assert(another);
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->device_type = LE_DEVICE_TYPE_CPU;
    self->element_type = another->element_type;
    self->shape = le_shape_copy(another->shape);
    self->stride = another->stride;
    self->owns_data = true;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    size_t data_size = elements_count * le_type_size(another->element_type);
    self->data = malloc(data_size);
    for (unsigned i = 0; i < elements_count; i++)
    {
        switch (self->element_type)
        {
        case LE_TYPE_INT8:
        case LE_TYPE_UINT8:
            ((int8_t *)self->data)[i] = 0;
            break;
        case LE_TYPE_INT16:
        case LE_TYPE_UINT16:
            ((int16_t *)self->data)[i] = 0;
            break;
        case LE_TYPE_INT32:
        case LE_TYPE_UINT32:
            ((int32_t *)self->data)[i] = 0;
            break;
        case LE_TYPE_FLOAT16:
            ((uint16_t *)self->data)[i] = F16_0;
            break;
        case LE_TYPE_FLOAT32:
            ((float *)self->data)[i] = 0.0f;
            break;
        case LE_TYPE_FLOAT64:
            ((double *)self->data)[i] = 0.0;
            break;
        default:
            break;
        }
    }
    return self;
}


LeTensor *
le_tensor_new_cast(LeTensor *another, LeType type)
{
    assert(another->device_type == LE_DEVICE_TYPE_CPU);
    assert(le_cast_rawcpy[type][another->element_type] || le_cast_fn[type][another->element_type]);
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->device_type = LE_DEVICE_TYPE_CPU;
    self->element_type = type;
    self->shape = le_shape_copy(another->shape);
    self->stride = another->stride;
    self->owns_data = true;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    size_t data_size = elements_count * le_type_size(self->element_type);
    self->data = malloc(data_size);
    
    if (le_cast_rawcpy[self->element_type][another->element_type])
    {
        assert(le_type_size(self->element_type) == le_type_size(another->element_type));
        memcpy(self->data, another->data, data_size);
    }
    else
    {
        for (unsigned i = 0; i < elements_count; i++)
        {
            le_cast_fn[self->element_type][another->element_type](self->data, another->data, i);
        }
    } 
    
    return self;
}

LeTensor *
le_tensor_new_equal_u8(LeType type, LeTensor *another, uint8_t scalar)
{
    assert(another->device_type == LE_DEVICE_TYPE_CPU);
    unsigned i;
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->device_type = LE_DEVICE_TYPE_CPU;
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
le_tensor_contiguous(const LeTensor *tensor)
{
    return tensor->stride == le_shape_get_size(tensor->shape, -1);
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
        
    if (!le_shape_equal(a->shape, b->shape))
        return false;
    
    assert(a->device_type == LE_DEVICE_TYPE_CPU);
    assert(b->device_type == LE_DEVICE_TYPE_CPU);
    
    uint32_t elements_count = le_shape_get_elements_count(a->shape);
    if (le_tensor_contiguous(a) && le_tensor_contiguous(b))
    {
        if (memcmp(a->data, b->data, elements_count * le_type_size(a->element_type)))
        {
            return false;
        }
    }
    else
    {
        /// @todo: Optimize case when both tensors are not contiguous but share same region size
        for (uint32_t i = 0; i < elements_count; i++)
        {
            if (memcmp(le_tensor_at(a, i), le_tensor_at(b, i), le_type_size(a->element_type)))
            {
                return false;
            }
        }
    }
    
    return true;
}

bool     
le_tensor_reshape(LeTensor *self, unsigned num_dimensions, ...)
{
    /// @todo: Take stride into account
    /// @todo: Add more assertions

    va_list args;
    va_start(args, num_dimensions);
    
    LeShape *new_shape = le_shape_new_uninitialized(num_dimensions);
    for (unsigned i = 0; i < num_dimensions; i++)
    {
        int size = va_arg(args, int);
        le_shape_set_size(new_shape, i, size);
    }

    va_end(args);
    
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
    
    assert(another->device_type == LE_DEVICE_TYPE_CPU);
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->device_type = LE_DEVICE_TYPE_CPU;
    self->element_type = another->element_type;
    self->shape = le_shape_lower_dimension(another->shape);
    self->stride = le_shape_get_size(self->shape, -1);

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
    
    assert(another->device_type == LE_DEVICE_TYPE_CPU);
    
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->device_type = LE_DEVICE_TYPE_CPU;
    self->element_type = another->element_type;
    self->shape = le_shape_lower_dimension(another->shape);
    self->stride = le_shape_get_size(self->shape, -1);
    
    size_t data_size = le_shape_get_elements_count(self->shape) * le_type_size(self->element_type);
    self->owns_data = true;
    self->data = malloc(data_size);
    
    memcpy(self->data, another->data + index * data_size, data_size);
    
    return self;
}

static inline uint32_t
virtual_index(uint32_t logical_index, uint32_t last_size, uint32_t stride)
{
    return (last_size == stride) ? logical_index : (logical_index / last_size * stride + logical_index % last_size);
}

void *
le_tensor_at(const LeTensor *tensor, uint32_t index)
{
    assert(tensor->device_type == LE_DEVICE_TYPE_CPU);
    
    return (uint8_t *)tensor->data + le_type_size(tensor->element_type) * virtual_index(index, le_shape_get_size(tensor->shape, -1), tensor->stride);
}

uint8_t
le_tensor_at_u8(const LeTensor *tensor, uint32_t index)
{
    assert(tensor->element_type == LE_TYPE_UINT8);
    assert(tensor->device_type == LE_DEVICE_TYPE_CPU);

    return ((uint8_t *)tensor->data)[virtual_index(index, le_shape_get_size(tensor->shape, -1), tensor->stride)];
}

uint32_t
le_tensor_at_u32(const LeTensor *tensor, uint32_t index)
{
    assert(tensor->element_type == LE_TYPE_UINT32);
    assert(tensor->device_type == LE_DEVICE_TYPE_CPU);

    return ((uint32_t *)tensor->data)[virtual_index(index, le_shape_get_size(tensor->shape, -1), tensor->stride)];
}

float
le_tensor_at_f32(const LeTensor *tensor, uint32_t index)
{
    assert(tensor->element_type == LE_TYPE_FLOAT32);
    assert(tensor->device_type == LE_DEVICE_TYPE_CPU);
    
    return ((float *)tensor->data)[virtual_index(index, le_shape_get_size(tensor->shape, -1), tensor->stride)];
}

void
le_tensor_assign(LeTensor *tensor, const LeTensor *another)
{
    assert(tensor->device_type == LE_DEVICE_TYPE_CPU);
    assert(another->device_type == LE_DEVICE_TYPE_CPU);
    
    if ((tensor->element_type == another->element_type)
        && (tensor->stride == le_shape_get_size(tensor->shape, -1))
        && (another->stride == le_shape_get_size(another->shape, -1))
        && le_shape_equal(tensor->shape, another->shape))
    {
        size_t data_size = le_shape_get_elements_count(another->shape) * le_type_size(another->element_type);
        memcpy(tensor->data, another->data, data_size);
    }
}

void
le_tensor_set_f32(LeTensor *tensor, uint32_t index, float value)
{
    assert(tensor->element_type == LE_TYPE_FLOAT32);
    assert(tensor->device_type == LE_DEVICE_TYPE_CPU);
    
    ((float *)tensor->data)[virtual_index(index, le_shape_get_size(tensor->shape, -1), tensor->stride)] = value;
}

void
le_matrix_empty(LeTensor *self)
{
    switch (self->device_type) {
    case LE_DEVICE_TYPE_CPU:
        free(self->data);
        self->data = NULL;
        break;
    default:
        assert(false);
        break;
    }
    le_shape_free(self->shape);
    self->shape = NULL;
    self->element_type = LE_TYPE_VOID;
}

float
le_dot_product(const LeTensor *a, const LeTensor *b)
{
    assert(a->device_type == LE_DEVICE_TYPE_CPU);
    assert(b->device_type == LE_DEVICE_TYPE_CPU);
    
#ifdef __APPLE__
    return le_accelerate_dot_product(a, b);
#elif defined(HAVE_OPENBLAS)
    return le_openblas_dot_product(a, b);
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
    assert(a->device_type == LE_DEVICE_TYPE_CPU);
    assert(b->device_type == LE_DEVICE_TYPE_CPU);
    
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
le_tensor_add_tensor(LeTensor *a, const LeTensor *b)
{
    /// @todo: Take stride into account
    assert(a->device_type == LE_DEVICE_TYPE_CPU);
    assert(b->device_type == LE_DEVICE_TYPE_CPU);
    assert(a->element_type == b->element_type);
    assert(le_shape_equal(a->shape, b->shape));
    
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(a->shape);
    
    switch (a->element_type)
    {
    case LE_TYPE_FLOAT32:
        for (i = 0; i < elements_count; i++)
        {
            ((float *)a->data)[i] += ((float *)b->data)[i];
        }
        break;
    case LE_TYPE_UINT32:
        for (i = 0; i < elements_count; i++)
        {
            ((uint32_t *)a->data)[i] += ((uint32_t *)b->data)[i];
        }
        break;
    default:
        assert(0);
        break;
    }
}

void
le_tensor_sub_f32(LeTensor *self, float b)
{
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
    assert(a->device_type == LE_DEVICE_TYPE_CPU);
    assert(b->device_type == LE_DEVICE_TYPE_CPU);
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
    assert(a->device_type == LE_DEVICE_TYPE_CPU);
    assert(b->device_type == LE_DEVICE_TYPE_CPU);
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
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
    switch (self->device_type)
    {
#ifdef HAVE_METAL
    case LE_DEVICE_TYPE_METAL:
        le_metal_tensor_mul_tensor(self, b);
        break;
#endif
#ifdef HAVE_CUDA
    case LE_DEVICE_TYPE_CUDA:
        le_cuda_tensor_mul_tensor(self, b);
        break;
#endif
    case LE_DEVICE_TYPE_CPU:
        for (unsigned i = 0, elements_count = le_shape_get_elements_count(self->shape);
             i < elements_count; i++)
        {
            ((float *)self->data)[i] *= ((float *)b->data)[i];
        }
        break;
    default:
        assert(false);
        break;
    }
}

void
le_tensor_div_u32(LeTensor *self, uint32_t b)
{
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
    assert(self->element_type == LE_TYPE_UINT32);

    /// @todo: Take stride into account
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        ((uint32_t *)self->data)[i] /= b;
    }

}

void
le_tensor_add_f32(LeTensor *self, float b)
{
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
    assert(self->element_type == LE_TYPE_FLOAT32);
    /// @todo: Take stride into account
    float sum = 0.0;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (unsigned i = 0; i < elements_count; i++)
    {
        sum += ((float *)self->data)[i];
    }
    
    return sum;
}

float
le_tensor_sad_f32(const LeTensor *a, const LeTensor *b)
{
    assert(a->device_type == LE_DEVICE_TYPE_CPU);
    assert(b->device_type == LE_DEVICE_TYPE_CPU);
    assert(a->element_type == LE_TYPE_FLOAT32);
    assert(b->element_type == LE_TYPE_FLOAT32);
    assert(le_shape_equal(a->shape, b->shape));

    float sad = 0.0;
    unsigned elements_count = le_shape_get_elements_count(a->shape);
    
    /// @note: SSE2 and ARM NEON provide instructions for this
    for (unsigned i = 0; i < elements_count; i++)
    {
        sad += fabs(le_tensor_at_f32(a, i) - le_tensor_at_f32(b, i));
    }
    
    return sad;

}

float
le_tensor_l2_f32(const LeTensor *tensor)
{
    assert(tensor->device_type == LE_DEVICE_TYPE_CPU);
    assert(tensor->element_type == LE_TYPE_FLOAT32);

    float l2 = 0.0;
    unsigned elements_count = le_shape_get_elements_count(tensor->shape);
    /// @todo: Speed up this
    for (unsigned i = 0; i < elements_count; i++)
    {
        float v = le_tensor_at_f32(tensor, i);
        l2 += v * v;
    }
    l2 = sqrtf(l2);
    
    return l2;
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
    switch (self->device_type) {
    case LE_DEVICE_TYPE_CPU:
#ifdef __APPLE__
        return le_accelerate_tensor_apply_sigmoid(self);
#else
        assert(self->element_type == LE_TYPE_FLOAT32);
        unsigned elements_count = le_shape_get_elements_count(self->shape);
        for (unsigned i = 0; i < elements_count; i++)
        {
            ((float *)self->data)[i] = le_sigmoid(((float *)self->data)[i]);
        }
#endif
        break;
#ifdef HAVE_CUDA
    case LE_DEVICE_TYPE_CUDA:
        le_cuda_tensor_apply_sigmoid(self);
        break;
#endif
#ifdef HAVE_METAL
    case LE_DEVICE_TYPE_METAL:
        le_metal_tensor_apply_sigmoid(self);
        break;
#endif
    default:
        assert(false);
        break;
    }
}

void
le_tensor_apply_sigmoid_prime(LeTensor *self)
{
    switch (self->device_type) {
    case LE_DEVICE_TYPE_CPU:
#ifdef __APPLE__
        return le_accelerate_tensor_apply_sigmoid_prime(self);
#else
        assert(self->element_type == LE_TYPE_FLOAT32);
        unsigned elements_count = le_shape_get_elements_count(self->shape);        
        for (unsigned i = 0; i < elements_count; i++)
        {
            float sigmoid = le_sigmoid(((float *)self->data)[i]);
            ((float *)self->data)[i] = sigmoid * (1.0f - sigmoid);
        }
#endif
        break;
#ifdef HAVE_CUDA
    case LE_DEVICE_TYPE_CUDA:
        le_cuda_tensor_apply_sigmoid_prime(self);
        break;
#endif
#ifdef HAVE_METAL
    case LE_DEVICE_TYPE_METAL:
        le_metal_tensor_apply_sigmoid_prime(self);
        break;
#endif
    default:
        assert(false);
        break;
    }
}

void
le_tensor_apply_tanh(LeTensor *self)
{
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
                    sprintf(ptr, "%f%n", ((float *)self->data)[y * self->shape->sizes[1] + x], &written);
                    break;
                case LE_TYPE_FLOAT64:
                    sprintf(ptr, "%lf%n", ((double *)self->data)[y * self->shape->sizes[1] + x], &written);
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
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
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
    {
        switch (self->device_type)
        {
        case LE_DEVICE_TYPE_CPU:
            free(self->data);
            break;

#ifdef HAVE_METAL
        case LE_DEVICE_TYPE_METAL:
            le_metal_data_free(self->data);
            break;
#endif
            
#ifdef HAVE_CUDA
        case LE_DEVICE_TYPE_CUDA:
            le_cuda_data_free(self->data);
            break;
#endif
            
        default:
            assert(false);
            break;
        }
    }
    
    le_shape_free(self->shape);
    free(self);
}

LeTensorStats
le_tensor_get_stats(LeTensor *self)
{
    assert(self->device_type == LE_DEVICE_TYPE_CPU);
    
    LeTensorStats stats;
    stats.deviation = 0.0f;
    stats.mean = 0.0f;
    stats.max = 0.0f;
    stats.min = 0.0f;
    stats.nans = 0;
    stats.zeros = 0;

    /// @todo: Take stride into account
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    uint32_t last_size = le_shape_get_size(self->shape, -1);

    if (elements_count >= 1)
    {
        float value = ((float *)self->data)[virtual_index(0, last_size, self->stride)];
        stats.max = value;
        stats.min = value;
        stats.mean = value;
        for (unsigned i = 1; i < elements_count; i++)
        {
            float value = ((float *)self->data)[virtual_index(i, last_size, self->stride)];
            if (value > stats.max)
                stats.max = value;
            if (value < stats.min)
                stats.min = value; 
            stats.mean += value;
        }
        stats.mean /= elements_count;
        for (unsigned i = 1; i < elements_count; i++)
        {
            float value = ((float *)self->data)[virtual_index(i, last_size, self->stride)];
            stats.deviation += fabs(value - stats.mean);
            if (isnan(value))
                stats.nans++;
            if (value == 0)
                stats.zeros = 0;
        }
        stats.deviation /= elements_count;
    }

    return stats;
}
