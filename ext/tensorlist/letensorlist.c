/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "tensorlist"

#include "letensorlist.h"
#include <stdio.h>
#include <stdint.h>
#include <le/letensor-imp.h>

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

        }
        fclose(fout);
    }
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
