/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stddef.h>

#ifndef __LEMEM_H__
#define __LEMEM_H__

void * le_alloc (size_t  size);

void   le_free  (void   *block);

#endif
