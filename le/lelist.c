/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include "lelist.h"

LeList *
le_list_last(LeList *list)
{
    if (list)
    {
        while (list->next)
        {
            list = list->next;
        }
    }
    
    return list;
}

LeList *
le_list_append(LeList *list, void *data)
{
    LeList *new_element = malloc(sizeof(struct LeList));
    new_element->data = data;
    new_element->next = NULL;
    
    if (list)
    {
        list = le_list_last(list);
        list->next = new_element;
        new_element->prev = list;
        return list;
    }
    else
    {
        new_element->prev = NULL;
        return new_element;
    }
}