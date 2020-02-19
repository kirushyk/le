/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "tensorlist"

#include "letensorlist.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <le/letensor-imp.h>

static void
le_tensor_serialize(LeTensor *tensor, FILE *fout)
{
    assert(tensor);
    assert(fout);

    fwrite((uint8_t *)&tensor->element_type, sizeof(uint8_t), 1, fout);
    fwrite((uint8_t *)&tensor->shape->num_dimensions, sizeof(uint8_t), 1, fout);
    fwrite(tensor->shape->sizes, sizeof(uint32_t), tensor->shape->num_dimensions, fout);
    unsigned elements_count = le_shape_get_elements_count(tensor->shape);
    fwrite(tensor->data, le_type_size(tensor->element_type), elements_count, fout);
}

void
le_tensorlist_save(LeList *tensors, const char *filename)
{
    FILE *fout = fopen(filename, "wb");
    if (fout)
    {
        uint8_t version = 1;
        fwrite(&version, sizeof(version), 1, fout);
        uint16_t num_tensors = 0;
        /// @todo: Make this be LeList function
        for (LeList *current = tensors; current != NULL; current = current->next)
            num_tensors++;
        fwrite(&num_tensors, sizeof(num_tensors), 1, fout);
        for (LeList *current = tensors; current != NULL; current = current->next)
        {
            le_tensor_serialize(LE_TENSOR(current->data), fout);
        }
        fclose(fout);
    }
}

static LeTensor *
le_tensor_deserialize(FILE *fin)
{
    assert(fin);

    LeTensor *self = malloc(sizeof(struct LeTensor));
    fread((uint8_t *)&self->element_type, sizeof(uint8_t), 1, fin);

    self->shape = malloc(sizeof(LeShape));
    fread((uint8_t *)&self->shape->num_dimensions, sizeof(uint8_t), 1, fin);
    self->shape->sizes = malloc(self->shape->num_dimensions * sizeof(uint32_t));
    fread(self->shape->sizes, sizeof(uint32_t), self->shape->num_dimensions, fin);

    if (self->shape->num_dimensions > 0)
        self->stride = le_shape_get_last_size(self->shape);
    else
        self->stride = 0;
    self->owns_data = true;
    unsigned elements_count = le_shape_get_elements_count(self->shape);
    self->data = malloc(elements_count * le_type_size(self->element_type));
    fread(self->data, le_type_size(self->element_type), elements_count, fin);
    
    return self;
}

LeList *
le_tensorlist_load(const char *filename)
{
    LeList *list = NULL;
    FILE *fin = fopen(filename, "rb");
    if (fin)
    {
        uint8_t version = 0;
        fread(&version, sizeof(version), 1, fin);
        if (version == 1)
        {
            uint16_t num_tensors = 0;
            fread(&num_tensors, sizeof(num_tensors), 1, fin);
            for (uint16_t i = 0; i < num_tensors; i++)
            {
                LeTensor *tensor = le_tensor_deserialize(fin);
                list = le_list_append(list, tensor);
            }
        }
        else
        {
            LE_WARNING("%s: Unknown version of .tensorlist file: %d", filename, (int)version);
        }
        fclose(fin);
    }
    else
    {
        LE_WARNING("File not found: %s", filename);
    }
    return list;
}
