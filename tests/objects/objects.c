/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>
#include "base.h"
#include "a.h"
#include "b.h"

int
main()
{
    Base *base = NULL;
    A *a = a_new();
    B *b = b_new();
    
    base = (Base *)a;
    assert(base->value == 13);
    assert(base_polymorphic(base) == 'a');
    
    base = (Base *)b;
    assert(base->value == 42);
    assert(base_polymorphic(base) == 'b');
    
    a_free(a);
    b_free(b);

    return EXIT_SUCCESS;
}
