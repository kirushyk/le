/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lelog.h"
#include <stdio.h>

void
le_log(const char *category, const char *message)
{
    fprintf(stderr, "[%s] %s: %s\n", category, "WARNING", message);
}
