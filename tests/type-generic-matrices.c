/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>
#include <le/letensor-imp.h>

int
main()
{
    LeTensor *m = le_tensor_new(LE_TYPE_FLOAT32, 2, 1, 1, 1.0);
    le_matrix_set(m, 0, 0, 2.0f);

    return EXIT_SUCCESS;
}
