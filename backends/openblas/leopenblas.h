/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LEOPENBLAS_H__
#define __LEOPENBLAS_H__

#include <le/le.h>

G_BEGIN_DECLS

LeTensor * le_openblas_matrix_new_product         (const LeTensor *a,
                                                   bool            transpose_a,
                                                   const LeTensor *b,
                                                   bool            transpose_b);

gfloat      le_openblas_dot_product                (const LeTensor *a,
                                                   const LeTensor *b);

G_END_DECLS

#endif
