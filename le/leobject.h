/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_OBJECT_H__
#define __LE_OBJECT_H__

typedef struct LeClass
{
    
} LeClass;

typedef struct LeObject
{
    struct LeClass *klass;
} LeObject;

LeObject * le_object_alloc (void);

void       le_object_free  (LeObject *);

#endif
