/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LELOG_H__
#define __LELOG_H__

#include <stdarg.h>

// #ifndef DEFAULT_LOG_CATEGORY
// #error DEFAULT_LOG_CATEGORY is not defined for current file
// #endif

typedef enum LeLogLevel
{
    LE_LOG_LEVEL_ERROR,
    LE_LOG_LEVEL_WARNING,
    LE_LOG_LEVEL_INFO,
    
    /// @note: Used to count amount of levels
    LE_LOG_LEVEL_LAST
} LeLogLevel;

/** @todo: Add Log Level parameter */
void le_log (const char *category,
             LeLogLevel  level,
             const char *message,
             ...);

#define LE_ERROR(...)   le_log(DEFAULT_LOG_CATEGORY, LE_LOG_LEVEL_ERROR,   __VA_ARGS__)
#define LE_WARNING(...) le_log(DEFAULT_LOG_CATEGORY, LE_LOG_LEVEL_WARNING, __VA_ARGS__)
#define LE_INFO(...)    le_log(DEFAULT_LOG_CATEGORY, LE_LOG_LEVEL_INFO,    __VA_ARGS__)

#endif
