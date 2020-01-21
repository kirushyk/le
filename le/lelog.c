/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lelog.h"
#include <stdio.h>
#include <stdlib.h>

static const char *le_level_name[(size_t)LE_LOG_LEVEL_LAST] =
{
    "ERROR",
    "WARNING",
    "INFO"
};

void
le_log(const char *category, LeLogLevel level, const char *message)
{
    fprintf(stderr, "[%s] %s: %s\n", category, le_level_name[(size_t)level], message);
}
