/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
 Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <stdlib.h>
#include "lelist.h"

LeList *
le_list_first(LeList *list)
{
    if (list)
    {
        while (list->prev)
        {
            list = list->prev;
        }
    }
    
    return list;
}

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
le_list_prepend(LeList *list, void *data)
{
    LeList *new_element = malloc(sizeof(struct LeList));
    new_element->data = data;
    new_element->prev = NULL;
    
    if (list)
    {
        LeList *first = le_list_first(list);
        first->prev = new_element;
        new_element->next = first;
    }
    else
    {
        new_element->next = NULL;
    }
    
    return new_element;
}

LeList *
le_list_append(LeList *list, void *data)
{
    LeList *new_element = malloc(sizeof(struct LeList));
    new_element->data = data;
    new_element->next = NULL;
    
    if (list)
    {
        LeList *last = le_list_last(list);
        last->next = new_element;
        new_element->prev = last;
        return list;
    }
    else
    {
        new_element->prev = NULL;
        return new_element;
    }
}

void
le_list_foreach(LeList *list, LeFunction function)
{
    while (list)
    {
        LeList *next = list->next;
        function(list->data);
        list = next;
    }
}

void
le_list_foreach2(LeList *list, LeCallback callback, void *user_data)
{
    while (list)
    {
        LeList *next = list->next;
        callback(list->data, user_data);
        list = next;
    }
}

void
le_list_free(LeList *list, LeFunction destroy)
{
    le_list_foreach(list, destroy);
    while (list)
    {
        LeList *next = list->next;
        next->prev = NULL;
        free(list);
        list = next;
    }
}
