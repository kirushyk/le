/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "sobel"

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>

int
main()
{
    LeTensor *output_image;
    LeTensor *expected_output;
    LeTensor *zeros = le_matrix_new_zeros(LE_TYPE_FLOAT32, 4, 4);

    LeTensor *sobel_gx_filter = le_tensor_new(LE_TYPE_FLOAT32, 2, 3, 3,   
        1.0, 2.0, 1.0,
        0.0, 0.0, 0.0,
        -1.0, -2.0, -1.0
    );
    LE_INFO("Horizontal Sobel Filter:\n%s", le_tensor_to_cstr(sobel_gx_filter));

    LeTensor *sobel_gy_filter = le_tensor_new(LE_TYPE_FLOAT32, 2, 3, 3,   
        1.0, 0.0, -1.0,
        2.0, 0.0, -2.0,
        1.0, 0.0, -1.0
    );
    LE_INFO("Vertical Sobel Filter:\n%s", le_tensor_to_cstr(sobel_gy_filter));

    LeTensor *vertical_edge_pattern = le_tensor_new(LE_TYPE_FLOAT32, 2, 6, 6,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0
    );
    LE_INFO("Vertical Edge Pattern");
    LE_INFO("Input Image:\n%s", le_tensor_to_cstr(vertical_edge_pattern));
    output_image = le_matrix_new_conv2d(vertical_edge_pattern, sobel_gx_filter);
    LE_INFO("Cross-correlation with Horizontal Sobel Filter:\n%s", le_tensor_to_cstr(output_image));
    assert(le_tensor_equal(output_image, zeros));
    le_tensor_unref(output_image);
    output_image = le_matrix_new_conv2d(vertical_edge_pattern, sobel_gy_filter);
    LE_INFO("Cross-correlation with Vertical Sobel Filter:\n%s", le_tensor_to_cstr(output_image));
    expected_output = le_tensor_new(LE_TYPE_FLOAT32, 2, 4, 4,
        0.0, 4.0, 4.0, 0.0,
        0.0, 4.0, 4.0, 0.0,
        0.0, 4.0, 4.0, 0.0,
        0.0, 4.0, 4.0, 0.0
    );
    assert(le_tensor_equal(output_image, expected_output));
    le_tensor_unref(expected_output);
    le_tensor_unref(output_image);
    le_tensor_unref(vertical_edge_pattern);

    LeTensor *horizontal_edge_pattern = le_tensor_new(LE_TYPE_FLOAT32, 2, 6, 6,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    );
    LE_INFO("Horizontal Edge Pattern");
    LE_INFO("Input Image:\n%s", le_tensor_to_cstr(horizontal_edge_pattern));
    output_image = le_matrix_new_conv2d(horizontal_edge_pattern, sobel_gx_filter);
    LE_INFO("Cross-correlation with Horizontal Sobel Filter:\n%s", le_tensor_to_cstr(output_image));
    expected_output = le_tensor_new(LE_TYPE_FLOAT32, 2, 4, 4,
        0.0, 0.0, 0.0, 0.0,
        4.0, 4.0, 4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        0.0, 0.0, 0.0, 0.0
    );
    assert(le_tensor_equal(output_image, expected_output));
    le_tensor_unref(expected_output);
    le_tensor_unref(output_image);
    output_image = le_matrix_new_conv2d(horizontal_edge_pattern, sobel_gy_filter);
    LE_INFO("Cross-correlation with Vertical Sobel Filter:\n%s", le_tensor_to_cstr(output_image));
    assert(le_tensor_equal(output_image, zeros));
    le_tensor_unref(output_image);
    le_tensor_unref(horizontal_edge_pattern);

    LeTensor *checkers_pattern = le_tensor_new(LE_TYPE_FLOAT32, 2, 6, 6,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0
    );
    LE_INFO("Checkers Pattern");
    LE_INFO("Input Image:\n%s", le_tensor_to_cstr(checkers_pattern));
    output_image = le_matrix_new_conv2d(checkers_pattern, sobel_gx_filter);
    LE_INFO("Cross-correlation with Horizontal Sobel Filter:\n%s", le_tensor_to_cstr(output_image));
    expected_output = le_tensor_new(LE_TYPE_FLOAT32, 2, 4, 4,
        0.0, 0.0, 0.0, 0.0,
        4.0, 2.0, -2.0, -4.0,
        4.0, 2.0, -2.0, -4.0,
        0.0, 0.0, 0.0, 0.0
    );
    assert(le_tensor_equal(output_image, expected_output));
    le_tensor_unref(expected_output);
    le_tensor_unref(output_image);
    output_image = le_matrix_new_conv2d(checkers_pattern, sobel_gy_filter);
    LE_INFO("Cross-correlation with Vertical Sobel Filter:\n%s", le_tensor_to_cstr(output_image));
    expected_output = le_tensor_new(LE_TYPE_FLOAT32, 2, 4, 4,
        0.0, 4.0, 4.0, 0.0,
        0.0, 2.0, 2.0, 0.0,
        0.0, -2.0, -2.0, 0.0,
        0.0, -4.0, -4.0, 0.0
    );
    assert(le_tensor_equal(output_image, expected_output));
    le_tensor_unref(expected_output);
    le_tensor_unref(output_image);
    le_tensor_unref(checkers_pattern);

    le_tensor_unref(sobel_gy_filter);
    le_tensor_unref(sobel_gx_filter);

    le_tensor_unref(zeros);

    return EXIT_SUCCESS;
}
