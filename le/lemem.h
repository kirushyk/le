/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEMEM_H__
#define __LEMEM_H__

#include <stddef.h>
#include "lemacros.h"

LE_BEGIN_DECLS

void *             le_alloc                                (size_t             size);

void               le_free                                 (void              *block);

LE_END_DECLS

#endif
