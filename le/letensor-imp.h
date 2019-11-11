/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEMATRIX_IMP_H__
#define __LEMATRIX_IMP_H__

#include <stdbool.h>
#include "letype.h"
#include "leshape.h"

struct LeTensor
{
    LeType   element_type;
    LeShape *shape;
    bool     owns_data;
    void    *data;
};

#endif

