/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <le/le.h>

int
main()
{
    LeTensor *sobel_filter = le_tensor_new(LE_TYPE_FLOAT32, 2, 3, 3,   
        1.0, 0.0, -1.0,
        2.0, 0.0, -2.0,
        1.0, 0.0, -1.0
    );

    LeTensor *image = le_tensor_new(LE_TYPE_FLOAT32, 2, 6, 6,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0
    );

    le_tensor_free(image);
    le_tensor_free(sobel_filter);

    return EXIT_SUCCESS;
}
