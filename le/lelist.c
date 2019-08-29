/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

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
    LeList *new_element = malloc();
    new_element->data = data;
    new_element->next = NULL;
    
    if (list)
    {
        LeList *list = le_list_last(list);
        list->next = new_element;
        new_element->prev = list;
        return list;
    }
    else
    {
        list->prev = NULL;
        return list;
    }
}
