/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include "leobject.h"
#include <stdlib.h>

struct LeObject
{
    
};

LeObject *
le_object_alloc(void)
{
    LeObject *object;
    object = malloc(sizeof(struct LeObject));
    return object;
}

void
le_object_free(LeObject *object)
{
    free(object);
}
