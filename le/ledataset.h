/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LETRAININGDATA_H__
#define __LETRAININGDATA_H__

#include "lematrix.h"

typedef struct LeDataSet LeDataSet;

LeDataSet *      le_data_set_new_copy   (LeMatrix  *input,
                                         LeMatrix  *output);

LeDataSet *      le_data_set_new_take   (LeMatrix  *input,
                                         LeMatrix  *output);

LeMatrix *       le_data_set_get_input  (LeDataSet *data);

LeMatrix *       le_data_set_get_output (LeDataSet *data);

void             le_data_set_free       (LeDataSet *data);

#endif
