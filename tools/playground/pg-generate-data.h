/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_PG_DATA_GENERATE__
#define __LE_PG_DATA_GENERATE__

#include <le/le.h>

LeTrainingData * pg_generate_data (const char *pattern, float negative_label);

#endif
