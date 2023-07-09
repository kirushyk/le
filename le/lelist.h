/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LELIST_H__
#define __LELIST_H__

#include "lemacros.h"

LE_BEGIN_DECLS

/// Doubly-Linked Lists - linked lists that can be iterated over in both directions
typedef struct LeList
{
    void *data;
    struct LeList *next;
    struct LeList *prev;
} LeList;

LeList *           le_list_first                           (LeList *                list);

LeList *           le_list_last                            (LeList *                list);

LeList *           le_list_prepend                         (LeList *                list,
                                                            void *                  data);

LeList *           le_list_append                          (LeList *                list,
                                                            void *                  data);

typedef void(* LeFunction)(void *element);
#define LE_FUNCTION(fn) ((LeFunction)(fn))

void               le_list_foreach                         (LeList *                list,
                                                            LeFunction              function);

typedef void(* LeCallback)(void *element, void *user_data);

void               le_list_foreach2                        (LeList *                list,
                                                            LeCallback              callback,
                                                            void *                  user_data);

void               le_list_free                            (LeList *                list,
                                                            LeFunction              destroy);

LE_END_DECLS

#endif
