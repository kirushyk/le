/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include <assert.h>
#include <le/le.h>

int
main()
{
    LeList *a = le_list_append(NULL, (void *)0x12345678);
    
    assert(a);
    assert(a->next == NULL);
    assert(a->prev == NULL);
    assert(a->data == (void *)0x12345678);
    
    LeList *b = le_list_append(a, (void *)0x87654321);
    
    assert(b);
    assert(b == a);
    assert(a->next != NULL);
    assert(a->prev == NULL);
    assert(a->data == (void *)0x12345678);
    
    LeList *c = le_list_last(b);
    
    assert(c);
    assert(c != a);
    assert(c->next == NULL);
    assert(c->prev == a);
    assert(c == a->next);
    assert(c->data == (void *)0x87654321);
    
    return EXIT_SUCCESS;
}
