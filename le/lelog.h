/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LELOG_H__
#define __LELOG_H__

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
             const char *message);

#define LE_ERROR(message)   le_log(DEFAULT_LOG_CATEGORY, LE_LOG_LEVEL_ERROR,   message)
#define LE_WARNING(message) le_log(DEFAULT_LOG_CATEGORY, LE_LOG_LEVEL_WARNING, message)
#define LE_INFO(message)    le_log(DEFAULT_LOG_CATEGORY, LE_LOG_LEVEL_INFO,    message)

#endif
