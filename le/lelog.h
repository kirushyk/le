/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LELOG_H__
#define __LELOG_H__

/** @todo: Add Log Level parameter */
void le_log (const char *category,
             const char *message);

#define LE_WARNING(message) le_log(DEFAULT_LOG_CATEGORY, message)

#endif
