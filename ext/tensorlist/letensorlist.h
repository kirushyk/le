/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LETENSORLIST_H__
#define __LETENSORLIST_H__

#include <le/le.h>

void     le_tensorlist_save (LeList     *tensors,
                             const char *filename);

LeList * le_tensorlist_load (const char *filename);

#endif
