/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "ledataset.h"
#include <stdlib.h>

struct LeDataSet
{
    LeTensor *x;
    LeTensor *y;
};

LeDataSet *
le_data_set_new_copy(LeTensor *x, LeTensor *y)
{
    LeDataSet *data = malloc(sizeof(LeDataSet));
    data->x = le_matrix_new_copy(x);
    data->y = le_matrix_new_copy(y);
    return data;
}

LeDataSet *
le_data_set_new_take(LeTensor *input, LeTensor *output)
{
    LeDataSet *data = malloc(sizeof(LeDataSet));
    data->x = input;
    data->y = output;
    return data;
}

LeTensor *
le_data_set_get_input(LeDataSet *data)
{
    return data->x;
}

LeTensor *
le_data_set_get_output(LeDataSet *data)
{
    return data->y;
}

void
le_data_set_free(LeDataSet *self)
{
    le_matrix_free(self->x);
    le_matrix_free(self->y);
    free(self);
}
