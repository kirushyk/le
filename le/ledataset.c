/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "ledataset.h"
#include <stdlib.h>

struct LeDataSet
{
    LeMatrix *x;
    LeMatrix *y;
};

LeDataSet *
le_data_set_new_copy(LeMatrix *x, LeMatrix *y)
{
    LeDataSet *data = malloc(sizeof(LeDataSet));
    data->x = le_matrix_new_copy(x);
    data->y = le_matrix_new_copy(y);
    return data;
}

LeDataSet *
le_data_set_new_take(LeMatrix *input, LeMatrix *output)
{
    LeDataSet *data = malloc(sizeof(LeDataSet));
    data->x = input;
    data->y = output;
    return data;
}

LeMatrix *
le_data_set_get_input(LeDataSet *data)
{
    return data->x;
}

LeMatrix *
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
