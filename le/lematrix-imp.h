/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEMATRIX_IMP_H_
#define __LEMATRIX_IMP_H_

struct LeMatrix
{
    float *data;
    
    /** @note: Temporary */
    unsigned width;
    unsigned height;
};

#endif

