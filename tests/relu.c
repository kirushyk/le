/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>

int
main()
{
    LeTensor *m;

    m = le_tensor_new(LE_TYPE_FLOAT32, 2, 1, 1, 1.0);
    le_matrix_set(m, 0, 0, 2.0f);
    assert(le_matrix_at_f32(m, 0, 0) == 2.0f);
    le_tensor_apply_relu(m);
    assert(le_matrix_at_f32(m, 0, 0) == 2.0f);
    le_matrix_set(m, 0, 0, -2.0f);
    assert(le_matrix_at_f32(m, 0, 0) == -2.0f);
    le_tensor_apply_relu(m);
    assert(le_matrix_at_f32(m, 0, 0) == 0);
    le_tensor_free(m);
    
    m = le_tensor_new(LE_TYPE_FLOAT64, 2, 1, 1, 1.0);
    le_matrix_set(m, 0, 0, 2.0);
    assert(le_matrix_at_f64(m, 0, 0) == 2.0);
    le_tensor_apply_relu(m);
    assert(le_matrix_at_f64(m, 0, 0) == 2.0);
    le_matrix_set(m, 0, 0, -2.0);
    assert(le_matrix_at_f64(m, 0, 0) == -2.0);
    le_tensor_apply_relu(m);
    assert(le_matrix_at_f64(m, 0, 0) == 0);
    le_tensor_free(m);

    m = le_tensor_new(LE_TYPE_INT8, 2, 1, 1, 1);
    le_matrix_set(m, 0, 0, (int8_t)2);
    assert(le_matrix_at_i8(m, 0, 0) == 2);
    le_tensor_apply_relu(m);
    assert(le_matrix_at_i8(m, 0, 0) == 2);
    le_matrix_set(m, 0, 0, (int8_t)-2);
    assert(le_matrix_at_i8(m, 0, 0) == -2);
    le_tensor_apply_relu(m);
    assert(le_matrix_at_i8(m, 0, 0) == 0);
    le_tensor_free(m);

    m = le_tensor_new(LE_TYPE_INT16, 2, 1, 1, 1);
    le_matrix_set(m, 0, 0, (int16_t)2);
    assert(le_matrix_at_i16(m, 0, 0) == 2);
    le_tensor_apply_relu(m);
    assert(le_matrix_at_i16(m, 0, 0) == 2);
    le_matrix_set(m, 0, 0, (int16_t)-2);
    assert(le_matrix_at_i16(m, 0, 0) == -2);
    le_tensor_apply_relu(m);
    assert(le_matrix_at_i16(m, 0, 0) == 0);
    le_tensor_free(m);

    m = le_tensor_new(LE_TYPE_INT32, 2, 1, 1, 1);
    le_matrix_set(m, 0, 0, (int32_t)2);
    assert(le_matrix_at_i32(m, 0, 0) == 2);
    le_tensor_apply_relu(m);
    assert(le_matrix_at_i32(m, 0, 0) == 2);
    le_matrix_set(m, 0, 0, (int32_t)-2);
    assert(le_matrix_at_i32(m, 0, 0) == -2);
    le_tensor_apply_relu(m);
    assert(le_matrix_at_i32(m, 0, 0) == 0);
    le_tensor_free(m);
    
    return EXIT_SUCCESS;
}
