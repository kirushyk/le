/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "tests/subtensor"

#include "test-config.h"
#include <stdlib.h>
#include <assert.h>
#include <le/le.h>

int
main()
{
    LeTensor *tensor = le_tensor_new(LE_TYPE_UINT32, 2, 3, 3,
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    );
    LeTensor *subtensor = le_tensor_pick(tensor, 0);
    le_tensor_free(subtensor);
    LeTensor *middle_column = le_matrix_get_column(tensor, 1);
    le_tensor_free(middle_column);
    le_tensor_free(tensor);
    return 0;
}
