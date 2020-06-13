/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lemem.h"
#include <stdlib.h>
#include <string.h>

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

char *
le_strdup(const char *str)
{
    size_t size = strlen(str) + 1;
    char *copy = le_alloc(size);
    if (copy)
    {
        memcpy(copy, str, size);
    }
    return copy;
}