/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "test-config.h"
#include <stdlib.h>
#include <assert.h>
#include <le/le.h>
#include <ext/tensorlist/letensorlist.h>

#define TENSORLIST_FILENAME TEST_DIR "/test.tensorlist"

int
main()
{
    LeList *tensorlist = le_list_append(NULL, le_tensor_new(LE_TYPE_UINT8, 2, 2, 2,
        1, 2,
        3, 4
    ));
    tensorlist = le_list_append(tensorlist, le_tensor_new(LE_TYPE_FLOAT32, 2, 3, 3,
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ));
    tensorlist = le_list_append(tensorlist, le_tensor_new(LE_TYPE_INT16, 2, 5, 5,
        5, 4, 3, 2, 1,
        6, 5, 4, 3, 2,
        7, 6, 5, 4, 3,
        8, 7, 6, 5, 4,
        9, 8, 7, 6, 5
    ));
    
    le_tensorlist_save(tensorlist, TENSORLIST_FILENAME);

    LeList *loaded_tensorlist = le_tensorlist_load(TENSORLIST_FILENAME);

    LeList *i = NULL, *j = NULL;
    for (i = tensorlist, j = loaded_tensorlist;
         i && j;
         i = i->next, j = j->next)
    {
        LeTensor *original = LE_TENSOR(i->data);
        LeTensor *loaded = LE_TENSOR(j->data);
        assert(le_tensor_equal(original, loaded));
    }

    assert(i == NULL);
    assert(j == NULL);

    return EXIT_SUCCESS;
}
