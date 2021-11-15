/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leidx.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#if defined(__linux__)
#include <byteswap.h>
#elif defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define bswap_32(x) OSSwapInt32(x)
#endif
#include <zlib.h>

struct IDXHeader
{
    uint16_t zeros;
    uint8_t type;
    uint8_t dimensionality;
};

LeTensor *
le_idx_read(const char *filename)
{
    LeTensor *tensor = NULL;
    FILE *fin = fopen(filename, "r");
    
    if (fin)
    {
        struct IDXHeader header;
        fread(&header, sizeof(struct IDXHeader), 1, fin);
        
        if (header.zeros == 0)
        {   
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

            int32_t shape_bytes[header.dimensionality];
            fread(shape_bytes, sizeof(int32_t), header.dimensionality, fin);

            LeShape *shape = le_shape_new_uninitialized(header.dimensionality);
            for (uint8_t i = 0; i < header.dimensionality; i++)
            {
                le_shape_set_size(shape, i, bswap_32(shape_bytes[i]));
            }
            
            size_t elements_count = 1;
            for (uint8_t i = 0; i < header.dimensionality; i++)
            {
                elements_count = elements_count * le_shape_get_size(shape, i);
            }

            size_t element_size = le_type_size(type);
                        
            tensor = le_tensor_new_uninitialized(type, shape);
            fread(le_tensor_get_data(tensor), element_size, elements_count, fin);
        }
        else
        {
            /// @note: Error in IDX file
        }
        
        fclose(fin);
    }
    
    return tensor;
}

LeTensor *
le_idx_gz_read(const char *filename)
{
    LeTensor *tensor = NULL;
    struct gzFile_s *fin = gzopen(filename, "r");
    
    if (fin)
    {
        struct IDXHeader header;
        gzread(fin, &header, sizeof(struct IDXHeader));
        
        if (header.zeros == 0)
        {   
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
            
            int32_t shape_bytes[header.dimensionality];
            gzread(fin, shape_bytes, sizeof(int32_t) * header.dimensionality);

            LeShape *shape = le_shape_new_uninitialized(header.dimensionality);
            for (uint8_t i = 0; i < header.dimensionality; i++)
            {
                le_shape_set_size(shape, i, bswap_32(shape_bytes[i]));
            }

            size_t elements_count = 1;
            for (uint8_t i = 0; i < header.dimensionality; i++)
            {
                elements_count = elements_count * le_shape_get_size(shape, i);
            }

            size_t element_size = le_type_size(type);
            
            tensor = le_tensor_new_uninitialized(type, shape);
            gzread(fin, le_tensor_get_data(tensor), (unsigned)(element_size * elements_count));
        }
        else
        {
            /// @note: Error in IDX file
        }
        
        gzclose(fin);
    }
    
    return tensor;
}
