/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#define DEFAULT_LOG_CATEGORY "sobel"

#include <stdlib.h>
#include <le/le.h>

int
main()
{
    LeTensor *input_image = le_tensor_new(LE_TYPE_FLOAT32, 2, 6, 6,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0
    );

    LE_INFO("input_image = %s", le_tensor_to_cstr(input_image));

    LeTensor *sobel_filter = le_tensor_new(LE_TYPE_FLOAT32, 2, 3, 3,   
        1.0, 0.0, -1.0,
        2.0, 0.0, -2.0,
        1.0, 0.0, -1.0
    );

    LE_INFO("sobel_filter = %s", le_tensor_to_cstr(sobel_filter));

    LeTensor *output_image = le_matrix_new_conv2d(input_image, sobel_filter);

    LE_INFO("output_image = %s", le_tensor_to_cstr(output_image));

    le_tensor_free(output_image);
    le_tensor_free(sobel_filter);
    le_tensor_free(input_image);

    return EXIT_SUCCESS;
}
