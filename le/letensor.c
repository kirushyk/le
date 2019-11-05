/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "letensor.h"
#include "lematrix-imp.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

LeTensor *
le_matrix_new(void)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->data = NULL;
    self->shape = NULL;
    self->element_type = LE_TYPE_VOID;
    return self;
}

LeTensor *
le_matrix_new_copy(LeTensor *another)
{
    LeTensor *self = malloc(sizeof(struct LeTensor));
    self->element_type = another->element_type;
    size_t data_size = le_shape_get_elements_count(another->shape) * le_type_size(self->element_type);
    self->data = malloc(data_size);
    memcpy(self->data, another->data, data_size);
    self->shape = le_shape_copy(another->shape);
    return self;
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

void
le_matrix_empty(LeTensor *self)
{
    free(self->data);
    self->data = NULL;
    free(self->shape);
    self->shape = NULL;
    self->element_type = LE_TYPE_VOID;
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

void
le_matrix_free(LeTensor *self)
{
    free(self->shape);
    free(self->data);
    free(self);
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
        return le_matrix_new();
        
    if (a->element_type != b->element_type)
        return le_matrix_new();
    
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
        result += a->data[y] * b->data[y];
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
        float sub = a->data[y] - b->data[y];
        result += sub * sub;
    }
    
    return expf(-result / (2.0f * sigma * sigma));
}

void
le_matrix_subtract(LeTensor *a, LeTensor *b)
{
    if (le_shape_equal(a->shape, b->shape))
    {
        unsigned i;
        unsigned elements_count = le_shape_get_elements_count(a->shape);
        
        for (i = 0; i < elements_count; i++)
        {
            a->data[i] -= b->data[i];
        }
    }
}

void
le_matrix_multiply_by_scalar(LeTensor *self, float b)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        self->data[i] *= b;
    }
}

void
le_matrix_add_scalar(LeTensor *self, float b)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        self->data[i] += b;
    }
}

float
le_matrix_sum(LeTensor *self)
{
    float sum = 0.0;
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        sum += self->data[i];
    }
    
    return sum;
}

static float
le_sigmoid(const float a)
{
    return 1.0 / (1.0 + exp(-a));
}

void
le_matrix_apply_sigmoid(LeTensor *self)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        self->data[i] = le_sigmoid(self->data[i]);
    }
}

void
le_matrix_apply_greater_than(LeTensor *self, float scalar)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        self->data[i] = self->data[i] > scalar ? 1.0f : 0.0f;
    }
}

void
le_matrix_apply_svm_prediction(LeTensor *self)
{
    unsigned i;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    
    for (i = 0; i < elements_count; i++)
    {
        self->data[i] = self->data[i] > 0.0f ? 1.0f : -1.0f;
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
            fprintf(stream, "%1.3f", self->data[y * self->shape->sizes[1] + x]);
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

LeTensor *
le_matrix_new_polynomia(LeTensor *a)
{
    int example;
    int feature, another_feature;
    int initial_features_count = le_matrix_get_height(a);
    int additional_features_count = initial_features_count * (initial_features_count + 1) / 2;
    int examples_count = le_matrix_get_width(a);
    
    LeTensor *polynomia = le_matrix_new_uninitialized(initial_features_count + additional_features_count, examples_count);
    for (example = 0; example < examples_count; example++)
    {
        for (feature = 0; feature < initial_features_count; feature++)
        {
            le_matrix_set_element(polynomia, feature, example, le_matrix_at(a, feature, example));
        }
        
        int additional_feature_index = initial_features_count;
        for (feature = 0; feature < initial_features_count; feature++)
        {
            for (another_feature = feature; another_feature < initial_features_count; another_feature++)
            {
                float additional_feature = le_matrix_at(a, feature, example) * le_matrix_at(a, another_feature, example);
                le_matrix_set_element(polynomia, additional_feature_index, example, additional_feature);
                additional_feature_index++;
            }
        }
    }
    
    return polynomia;
}
