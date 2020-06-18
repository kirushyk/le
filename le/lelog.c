/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "lelog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <le/lelist.h>

static const char *le_level_name[(size_t)LE_LOG_LEVEL_LAST] =
{
    "ERROR",
    "WARNING",
    "INFO"
};

bool
category_present(const char *category)
{
    /// @todo: Use secure_getenv
    char *le_debug = getenv("LE_DEBUG");
    if (le_debug == NULL)
        return true;

    static bool categories_parsed = false;
    static LeList *categories_requested = NULL;
    if (!categories_parsed)
    {
        char *sep = ",";
        char *word, *brkt;

        for (word = strtok_r(le_debug, sep, &brkt);
             word;
             word = strtok_r(NULL, sep, &brkt))
        {
            categories_requested = le_list_prepend(categories_requested, word);
        }

        categories_parsed = true;
    }
    for (LeList *categories_iterator = categories_requested;
         categories_iterator;
         categories_iterator = categories_iterator->next)
    {
        if (strcmp((char *)categories_iterator->data, category) == 0)
            return true;
    }
    return false;
}

void
le_log(const char *category, LeLogLevel level, const char *message, ...)
{
    if (level == LE_LOG_LEVEL_INFO)
    {
        if (!category_present(category))
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
