/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "pg-generate-data.h"
#include <glib.h>
#include <math.h>

LeDataSet *
pg_generate_data(const char *pattern_name, unsigned examples_count)
{
    LeTensor *input = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, 2, examples_count);
    LeTensor *output = le_matrix_new_rand_f32(LE_DISTRIBUTION_UNIFORM, 1, examples_count);
    
    if (g_strcmp0(pattern_name, "spiral") == 0)
    {
        for (unsigned i = 0; i < examples_count; i++)
        {
            gfloat scalar = (rand() * 2.0f / RAND_MAX) - 1.0f;
            gfloat x = sinf(scalar * 3.0f * M_PI) * fabs(scalar);
            gfloat y = cosf(scalar * 3.0f * M_PI) * scalar;
            le_matrix_set(input, 0, i, x);
            le_matrix_set(input, 1, i, y);
            le_matrix_set(output, 0, i, scalar > 0.0f ? 1.0f : 0.0f);
        }
    }
    else if (g_strcmp0(pattern_name, "nested") == 0)
    {
        for (unsigned i = 0; i < examples_count; i++)
        {
            gfloat distance = (gfloat)rand() / RAND_MAX;
            gfloat angle = rand() * 2.0f * M_PI / RAND_MAX;
            gfloat x = sinf(angle) * distance;
            gfloat y = cosf(angle) * distance;
            le_matrix_set(input, 0, i, x);
            le_matrix_set(input, 1, i, y);
            le_matrix_set(output, 0, i, distance < 0.5f ? 1.0f : 0.0f);
        }
    }
    else if (g_strcmp0(pattern_name, "linsep") == 0)
    {
        gfloat bias = (gfloat)rand() / RAND_MAX - 0.5f;
        gfloat slope = rand() * 20.0f / RAND_MAX - 10.0f;
        le_tensor_mul(input, 2.0f);
        le_tensor_add(input, -1.0f);
        for (unsigned i = 0; i < examples_count; i++)
        {
            gfloat x = le_matrix_at_f32(input, 0, i);
            gfloat y = le_matrix_at_f32(input, 1, i);
            
            le_matrix_set(output, 0, i, (y > bias + slope * x) ? 1.0f : 0.0f);
        }
    }
    else if (g_strcmp0(pattern_name, "svb") == 0)
    {
#define SUPPORT_VECTORS_COUNT 4
        gfloat svx[SUPPORT_VECTORS_COUNT], svy[SUPPORT_VECTORS_COUNT];
        
        for (unsigned j = 0; j < SUPPORT_VECTORS_COUNT; j++)
        {
            svx[j] = (rand() * 2.0f / RAND_MAX) - 1.0f;
            svy[j] = (rand() * 2.0f / RAND_MAX) - 1.0f;
        }
        
        le_tensor_mul(input, 2.0f);
        le_tensor_add(input, -1.0f);
        for (unsigned i = 0; i < examples_count; i++)
        {
            unsigned closest_vector = 0;
            gfloat min_squared_distance = 2.0f;
            
            gfloat x = le_matrix_at_f32(input, 0, i);
            gfloat y = le_matrix_at_f32(input, 1, i);
            
            for (unsigned j = 0; j < SUPPORT_VECTORS_COUNT; j++)
            {
                gfloat squared_distance = (x - svx[j]) * (x - svx[j]) + (y - svy[j]) * (y - svy[j]);
                if (squared_distance < min_squared_distance)
                {
                    min_squared_distance = squared_distance;
                    closest_vector = j;
                }
            }
            
            le_matrix_set(output, 0, i, (closest_vector >= SUPPORT_VECTORS_COUNT / 2) ? 1.0f : 0.0f);
        }
    }
    else
    {
        le_tensor_mul(input, 2.0f);
        le_tensor_add(input, -1.0f);
    }
    
    return le_data_set_new_take(input, output);
}
