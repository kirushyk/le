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

    LeTensor *subtensor = le_tensor_pick(tensor, 1);
    assert(le_tensor_at_u32(subtensor, 0) == 4);
    assert(le_tensor_at_u32(subtensor, 1) == 5);
    assert(le_tensor_at_u32(subtensor, 2) == 6);
    LeTensor *subtensor_copy = le_tensor_new_copy(subtensor);
    assert(le_tensor_at_u32(subtensor_copy, 0) == 4);
    assert(le_tensor_at_u32(subtensor_copy, 1) == 5);
    assert(le_tensor_at_u32(subtensor_copy, 2) == 6);
    assert(le_tensor_equal(subtensor, subtensor_copy));
    LeTensor *subtensor_direct_copy = le_tensor_pick_copy(subtensor, 1);
    assert(le_tensor_at_u32(subtensor_direct_copy, 0) == 4);
    assert(le_tensor_at_u32(subtensor_direct_copy, 1) == 5);
    assert(le_tensor_at_u32(subtensor_direct_copy, 2) == 6);
    assert(le_tensor_equal(subtensor, subtensor_direct_copy));
    assert(le_tensor_equal(subtensor_direct_copy, subtensor_copy));
    le_tensor_free(subtensor_direct_copy);
    le_tensor_free(subtensor_copy);
    le_tensor_free(subtensor);

    LeTensor *middle_column = le_matrix_get_column(tensor, 1);
    assert(le_matrix_at_u32(middle_column, 0, 0) == 2);
    assert(le_matrix_at_u32(middle_column, 1, 0) == 5);
    assert(le_matrix_at_u32(middle_column, 2, 0) == 8);
    assert(le_tensor_at_u32(middle_column, 0) == 2);
    assert(le_tensor_at_u32(middle_column, 1) == 5);
    assert(le_tensor_at_u32(middle_column, 2) == 8);
    LeTensor *middle_column_copy = le_tensor_new_copy(middle_column);
    assert(le_matrix_at_u32(middle_column_copy, 0, 0) == 2);
    assert(le_matrix_at_u32(middle_column_copy, 1, 0) == 5);
    assert(le_matrix_at_u32(middle_column_copy, 2, 0) == 8);
    assert(le_tensor_at_u32(middle_column_copy, 0) == 2);
    assert(le_tensor_at_u32(middle_column_copy, 1) == 5);
    assert(le_tensor_at_u32(middle_column_copy, 2) == 8);
    assert(le_tensor_equal(middle_column, middle_column_copy));
    LeTensor *middle_column_direct_copy = le_matrix_get_column(tensor, 1);
    assert(le_matrix_at_u32(middle_column_direct_copy, 0, 0) == 2);
    assert(le_matrix_at_u32(middle_column_direct_copy, 1, 0) == 5);
    assert(le_matrix_at_u32(middle_column_direct_copy, 2, 0) == 8);
    assert(le_tensor_at_u32(middle_column_direct_copy, 0) == 2);
    assert(le_tensor_at_u32(middle_column_direct_copy, 1) == 5);
    assert(le_tensor_at_u32(middle_column_direct_copy, 2) == 8);
    assert(le_tensor_equal(middle_column, middle_column_direct_copy));
    assert(le_tensor_equal(middle_column_copy, middle_column_direct_copy));
    le_tensor_free(middle_column_direct_copy);
    le_tensor_free(middle_column_copy);
    le_tensor_free(middle_column);

    le_tensor_free(tensor);
    return 0;
}
