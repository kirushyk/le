/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leidx.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#if defined(__linux__)
#include <byteswap.h>
#define bswap_32(x) __bswap_32(x)
#elif defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define bswap_32(x) OSSwapInt32(x)
#endif

struct IDXHeader
{
    uint16_t zeros;
    uint8_t type;
    uint8_t dimensionality;
};

LeTensor * le_idx_read(const char *filename)
{
    LeTensor *tensor = NULL;
    FILE *fin = fopen(filename, "r");
    
    if (fin)
    {
        struct IDXHeader header;
        
        fread(&header, sizeof(struct IDXHeader), 1, fin);
        
        if (header.zeros == 0)
        {
            uint32_t *sizes = malloc(header.dimensionality * sizeof(uint32_t));
            fread(sizes, sizeof(uint32_t), header.dimensionality, fin);
            for (uint8_t i = 0; i < header.dimensionality; i++)
            {
                sizes[i] = bswap_32(sizes[i]);
            }
            
            size_t elements_count = 1;
            
            LeType type = LE_TYPE_VOID;
            
            switch (header.type) {
            case 0x08:
                type = LE_TYPE_UINT8;
                break;
                    
            case 0x09:
                type = LE_TYPE_INT8;
                break;
                    
            case 0x0B:
                type = LE_TYPE_INT16;
                break;

            case 0x0C:
                type = LE_TYPE_INT32;
                break;
                        
            case 0x0D:
                type = LE_TYPE_FLOAT32;
                break;

            case 0x0E:
                type = LE_TYPE_FLOAT64;
                break;

            default:
                /// @note: Error in IDX file
                type = LE_TYPE_VOID;
                break;
            }
            
            for (uint8_t i = 0; i < header.dimensionality; i++)
            {
                elements_count = elements_count * sizes[i];
            }

            size_t element_size = le_type_size(type);
            uint8_t *data = malloc(element_size * elements_count);
            fread(data, element_size, elements_count, fin);
            free(sizes);
            
            LeShape *shape = le_shape_new(0);
            
            tensor = le_tensor_new_from_data(type, shape, data);
        }
        else
        {
            /// @note: Error in IDX file
        }
        
        fclose(fin);
    }
    
    return tensor;
}
