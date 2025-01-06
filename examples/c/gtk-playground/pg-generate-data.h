/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __GTK_PLAYGROUND_PG_GENERATE_DATA_H__
#define __GTK_PLAYGROUND_PG_GENERATE_DATA_H__

#include <le/le.h>
#include <ext/simple-dataset/ledataset.h>

LeDataSet * pg_generate_data (const gchar *pattern, guint examples_count);

#endif
