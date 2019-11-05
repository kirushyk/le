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
    LeTensor *matrix = NULL;
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
            
            size_t element_size = 1;
            size_t elements_count = 1;
            
            switch (header.type) {
            case 0x08:
            case 0x09:
                element_size = 1;
                break;
                    
            case 0x0B:
                element_size = 2;
                break;

            case 0x0C:
            case 0x0D:
                element_size = 4;
                break;

            case 0x0E:
                element_size = 8;
                break;

            default:
                /// @note: Error in IDX file
                break;
            }
            
            for (uint8_t i = 0; i < header.dimensionality; i++)
            {
                elements_count = elements_count * sizes[i];
            }

            uint8_t *data = malloc(element_size * elements_count);
            fread(data, element_size, elements_count, fin);
            free(sizes);
//            0x08: unsigned byte
//            0x09: signed byte
//            0x0B: short (2 bytes)
//            0x0C: int (4 bytes)
//            0x0D: float (4 bytes)
//            0x0E: double (8 bytes)
        }
        else
        {
            /// @note: Error in IDX file
        }
        
        fclose(fin);
    }
    
    return matrix;
}
