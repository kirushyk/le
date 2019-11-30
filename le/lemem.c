/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lemem.h"
#include <stdlib.h>

void *
le_alloc(size_t size)
{
    return malloc(size);
}

void
le_free(void *block)
{
    free(block);
}
