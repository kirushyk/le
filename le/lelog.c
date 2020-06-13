/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lelog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *le_level_name[(size_t)LE_LOG_LEVEL_LAST] =
{
    "ERROR",
    "WARNING",
    "INFO"
};

void
le_log(const char *category, LeLogLevel level, const char *message, ...)
{
    if (level == LE_LOG_LEVEL_INFO)
    {
        const char *le_debug = getenv("LE_DEBUG");
        if ((le_debug == NULL) || strcmp(le_debug, category))
        {
            return;
        }
    }
    FILE *out = (level == LE_LOG_LEVEL_ERROR) ? stderr : stdout;
    fprintf(out, "[%s] %s: ", category, le_level_name[(size_t)level]);
    va_list args;
    va_start(args, message);
    vfprintf(out, message, args);
    va_end(args);
    fputc('\n', out);
    if (level == LE_LOG_LEVEL_ERROR)
    {
        exit(EXIT_FAILURE);
    }
}
