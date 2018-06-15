/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LE_TESTS_OBJECTS_A_H__
#define __LE_TESTS_OBJECTS_A_H__

typedef struct A A;

A *  a_new  (void);

void a_free (A *a);

#endif /* _LE_TESTS_OBJECTS_A_H_ */
