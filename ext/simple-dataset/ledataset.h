/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LETRAININGDATA_H__
#define __LETRAININGDATA_H__

#include <le/tensors/letensor.h>
#include "glib.h"

G_BEGIN_DECLS

typedef struct LeDataSet LeDataSet;

LeDataSet *      le_data_set_new_copy   (LeTensor  *input,
                                         LeTensor  *output);

LeDataSet *      le_data_set_new_take   (LeTensor  *input,
                                         LeTensor  *output);

LeTensor *       le_data_set_get_input  (LeDataSet *data);

LeTensor *       le_data_set_get_output (LeDataSet *data);

void             le_data_set_free       (LeDataSet *data);

G_END_DECLS

#endif
