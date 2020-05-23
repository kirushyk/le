/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_OBJECT_H__
#define __LE_OBJECT_H__

#include "lemacros.h"

LE_BEGIN_DECLS

typedef struct LeObject LeObject;

typedef struct LeClass
{
    void (*destructor)(LeObject *model);
} LeClass;

struct LeObject
{
    struct LeClass *klass;
};

#define LE_OBJECT(obj) ((LeObject *)(obj))
#define LE_CLASS(klass) ((LeClass *)(klass))
#define LE_OBJECT_GET_CLASS(obj) (LE_OBJECT(obj)->klass)

LeObject * le_object_alloc (void);

void       le_object_free  (LeObject *);

LE_END_DECLS

#endif
