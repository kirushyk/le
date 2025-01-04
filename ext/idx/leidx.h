/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <le/le.h>

#ifndef __EXT_IDX_LEIDX_H__
#define __EXT_IDX_LEIDX_H__

LeTensor * le_idx_read    (const char * filename);

LeTensor * le_idx_gz_read (const char * filename);

#endif
