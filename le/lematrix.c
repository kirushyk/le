/**
  *
  */

#include "lematrix.h"
#include "lematrix-imp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

LeMatrix *
le_matrix_new(void)
{
    LeMatrix *self = malloc(sizeof(struct LeMatrix));
    self->data = NULL;
    self->height = 0;
    self->width = 0;
    return self;
}

LeMatrix *
le_matrix_new_copy(LeMatrix *another)
{
    LeMatrix *self = malloc(sizeof(struct LeMatrix));
    self->data = malloc(another->height * another->width * sizeof(float));
    self->height = another->height;
    self->width = another->width;
    return self;
}

LeMatrix *
le_matrix_new_from_data(unsigned height, unsigned width, const float *data)
{
    LeMatrix *self;
    size_t data_size = height * width * sizeof(float);
    
    self = malloc(sizeof(struct LeMatrix));
    self->data = malloc(data_size);
    self->height = height;
    self->width = width;
    memcpy(self->data, data, data_size);
    
    return self;
}

unsigned
le_matrix_get_width(LeMatrix *self)
{
    return self->width;
}

unsigned
le_matrix_get_height(LeMatrix *self)
{
    return self->height;
}

float
le_matrix_at(LeMatrix *self, unsigned y, unsigned x)
{
    return self->data[y * self->width + x];
}

void
le_matrix_set_element(LeMatrix *self, unsigned y, unsigned x, float value)
{
    self->data[y * self->width + x] = value;
}

void
le_matrix_empty(LeMatrix *self)
{
    free(self->data);
    self->data = NULL;
    self->width = 0;
    self->height = 0;
}

LeMatrix *
le_matrix_new_identity(unsigned size)
{
    unsigned x;
    unsigned y;
    LeMatrix *self;
    
    self = malloc(sizeof(struct LeMatrix));
    self->data = malloc(size * size * sizeof(float));
    self->height = size;
    self->width = size;
    
    for (y = 0; y < size; y++)
    {
        for (x = 0; x < size; x++)
        {
            self->data[y * self->width + x] = (x == y) ? 1.0 : 0.0;
        }
    }
    
    return self;
}

LeMatrix *
le_matrix_new_zeros(unsigned height, unsigned width)
{
    unsigned i;
    unsigned elements_count;
    LeMatrix *self;
    
    self = malloc(sizeof(struct LeMatrix));
    self->data = malloc(height * width * sizeof(float));
    self->height = height;
    self->width = width;
    elements_count = height * width;
    
    for (i = 0; i < elements_count; i++)
    {
        self->data[i] = 0.0;
    }
    
    return self;
}


LeMatrix *
le_matrix_new_rand(unsigned height, unsigned width)
{
    unsigned x;
    unsigned y;
    LeMatrix *self;
    
    self = malloc(sizeof(struct LeMatrix));
    self->data = malloc(height * width * sizeof(float));
    self->height = height;
    self->width = width;
    
    for (y = 0; y < self->height; y++)
    {
        for (x = 0; x < self->width; x++)
        {
            self->data[y * self->width + x] = rand() / (float)RAND_MAX;
        }
    }
    
    return self;
}

void
le_matrix_free(LeMatrix *self)
{
    free(self->data);
    free(self);
}

LeMatrix *
le_matrix_new_transpose(LeMatrix *a)
{
    unsigned x;
    unsigned y;
    LeMatrix *self;
    
    self = malloc(sizeof(struct LeMatrix));
    self->data = malloc(a->width * a->height * sizeof(float));
    self->height = a->width;
    self->width = a->height;
    
    for (y = 0; y < self->height; y++)
    {
        for (x = 0; x < self->width; x++)
        {
            self->data[y * self->width + x] = a->data[x * a->width + y];
        }
    }
    
    return self;
}

LeMatrix *
le_matrix_new_product(LeMatrix *a, LeMatrix *b)
{
    unsigned x;
    unsigned y;
    unsigned i;
    
    LeMatrix *self;
    
    if (a->width != b->height)
        return le_matrix_new();
    
    self = malloc(sizeof(struct LeMatrix));
    self->data = malloc(a->height * b->width * sizeof(float));
    self->height = a->height;
    self->width = b->width;
    
    for (y = 0; y < self->height; y++)
    {
        for (x = 0; x < self->width; x++)
        {
            self->data[y * self->width + x] = 0.0f;
            for (i = 0; i < a->width; i++)
            {
                self->data[y * self->width + x] += a->data[y * a->width + i] * b->data[i * b->width + x];
            }
        }
    }
    
    return self;
}

void
le_matrix_subtract(LeMatrix *a, LeMatrix *b)
{
    if (a->height == b->height && a->width == b->width)
    {
        unsigned i;
        unsigned elements_count = a->height * a->width;
        
        for (i = 0; i < elements_count; i++)
        {
            a->data[i] -= b->data[i];
        }
    }
}

void
le_matrix_multiply_by_scalar(LeMatrix *self, float b)
{
    unsigned i;
    unsigned elements_count = self->height * self->width;
    
    for (i = 0; i < elements_count; i++)
    {
        self->data[i] *= b;
    }
}

void
le_matrix_add_scalar(LeMatrix *self, float b)
{
    unsigned i;
    unsigned elements_count = self->height * self->width;
    
    for (i = 0; i < elements_count; i++)
    {
        self->data[i] += b;
    }
}

float
le_matrix_sum(LeMatrix *self)
{
    float sum = 0.0;
    unsigned i;
    unsigned elements_count = self->height * self->width;
    
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
le_matrix_apply_sigmoid(LeMatrix *self)
{
    unsigned i;
    unsigned elements_count = self->height * self->width;
    
    for (i = 0; i < elements_count; i++)
    {
        self->data[i] = le_sigmoid(self->data[i]);
    }
}

/** @note: Temporary */
void
le_matrix_print(LeMatrix *self, FILE *stream)
{
    unsigned x;
    unsigned y;
    fprintf(stream, "[");
    for (y = 0; y < self->height; y++)
    {
        for (x = 0; x < self->width; x++)
        {
            fprintf(stream, "%1.3f", self->data[y * self->width + x]);
            if (x < self->width - 1)
            {
                fprintf(stream, " ");
            }
        }
        if (y < self->height - 1)
        {
            fprintf(stream, ";\n ");
        }
    }
    fprintf(stream, "]\n");
}
