/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_TESTS_OBJECTS_BASE_H__
#define __LE_TESTS_OBJECTS_BASE_H__

#include <le/le.h>

typedef struct Base
{
    LeObject parent;
    int value;
} Base;

typedef struct BaseClass
{
    LeClass parent;
    char (*base_polymorphic)(Base *base);
} BaseClass;

void base_construct   (Base *base);

char base_polymorphic (Base *base);

#endif /* _LE_TESTS_OBJECTS_BASE_H_ */
