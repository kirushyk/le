/**
  *
  */

#include "lematrix.h"
#include "lematrix-imp.h"
#include <stdlib.h>
#include <string.h>

LeMatrix *
le_matrix_new(void)
{
    LeMatrix *self = malloc(sizeof(struct LeMatrix));
    self->data = NULL;
    self->width = 0;
    self->height = 0;
    return self;
}

LeMatrix *
le_matrix_new_from_data(unsigned height, unsigned width, const double *data)
{
    LeMatrix *self;
    size_t data_size = height * width * sizeof(double);
    
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

double
le_matrix_at(LeMatrix *self, unsigned y, unsigned x)
{
    return self->data[y * self->width + x];
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
    self->data = malloc(size * size * sizeof(double));
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
    self->data = malloc(height * width * sizeof(double));
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
    self->data = malloc(height * width * sizeof(double));
    self->height = height;
    self->width = width;
    
    for (y = 0; y < self->height; y++)
    {
        for (x = 0; x < self->width; x++)
        {
            self->data[y * self->width + x] = rand() / (double)RAND_MAX;
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
    self->data = malloc(a->width * a->height * sizeof(double));
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
    self->data = malloc(a->height * b->width * sizeof(double));
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
le_matrix_add_scalar(LeMatrix *self, double b)
{
    unsigned i;
    unsigned elements_count = self->height * self->width;
    
    for (i = 0; i < elements_count; i++)
    {
        self->data[i] += b;
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
            fprintf(stream, "%1.3lf", self->data[y * self->width + x]);
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
