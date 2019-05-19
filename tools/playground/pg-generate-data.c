/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "pg-generate-data.h"
#include <glib.h>
#include <math.h>

LeTrainingData *
pg_generate_data(const char *pattern_name, unsigned examples_count)
{
    LeMatrix *input = le_matrix_new_rand(2, examples_count);
    LeMatrix *output = le_matrix_new_rand(1, examples_count);
    
    if (g_strcmp0(pattern_name, "spiral") == 0)
    {
        unsigned i;
        for (i = 0; i < examples_count; i++)
        {
            float scalar = (rand() * 2.0f / RAND_MAX) - 1.0f;
            float x = sinf(scalar * 3.0f * M_PI) * fabs(scalar);
            float y = cosf(scalar * 3.0f * M_PI) * scalar;
            le_matrix_set_element(input, 0, i, x);
            le_matrix_set_element(input, 1, i, y);
            le_matrix_set_element(output, 0, i, scalar > 0.0f ? 1.0f : 0.0f);
        }
    }
    else if (g_strcmp0(pattern_name, "nested") == 0)
    {
        unsigned i;
        for (i = 0; i < examples_count; i++)
        {
            float distance = (float)rand() / RAND_MAX;
            float angle = rand() * 2.0f * M_PI / RAND_MAX;
            float x = sinf(angle) * distance;
            float y = cosf(angle) * distance;
            le_matrix_set_element(input, 0, i, x);
            le_matrix_set_element(input, 1, i, y);
            le_matrix_set_element(output, 0, i, distance < 0.5f ? 1.0f : 0.0f);
        }
    }
    else if (g_strcmp0(pattern_name, "linsep") == 0)
    {
        unsigned i;
        float bias = (float)rand() / RAND_MAX - 0.5f;
        float slope = rand() * 20.0f / RAND_MAX - 10.0f;
        le_matrix_multiply_by_scalar(input, 2.0f);
        le_matrix_add_scalar(input, -1.0f);
        for (i = 0; i < examples_count; i++)
        {
            float x = le_matrix_at(input, 0, i);
            float y = le_matrix_at(input, 1, i);
            
            le_matrix_set_element(output, 0, i, (y > bias + slope * x) ? 1.0f : 0.0f);
        }
    }
    else if (g_strcmp0(pattern_name, "svb") == 0)
    {
        unsigned i, j;
        
#define SUPPORT_VECTORS_COUNT 4
        float svx[SUPPORT_VECTORS_COUNT], svy[SUPPORT_VECTORS_COUNT];
        
        for (j = 0; j < SUPPORT_VECTORS_COUNT; j++)
        {
            svx[j] = (rand() * 2.0f / RAND_MAX) - 1.0f;
            svy[j] = (rand() * 2.0f / RAND_MAX) - 1.0f;
        }
        
        le_matrix_multiply_by_scalar(input, 2.0f);
        le_matrix_add_scalar(input, -1.0f);
        for (i = 0; i < examples_count; i++)
        {
            unsigned closest_vector = 0;
            float min_squared_distance = 2.0f;
            
            float x = le_matrix_at(input, 0, i);
            float y = le_matrix_at(input, 1, i);
            
            for (j = 0; j < SUPPORT_VECTORS_COUNT; j++)
            {
                float squared_distance = (x - svx[j]) * (x - svx[j]) + (y - svy[j]) * (y - svy[j]);
                if (squared_distance < min_squared_distance)
                {
                    min_squared_distance = squared_distance;
                    closest_vector = j;
                }
            }
            
            le_matrix_set_element(output, 0, i, (closest_vector >= SUPPORT_VECTORS_COUNT / 2) ? 1.0f : 0.0f);
        }
    }
    else
    {
        le_matrix_multiply_by_scalar(input, 2.0f);
        le_matrix_add_scalar(input, -1.0f);
    }
    
    return le_training_data_new_take(input, output);
}
