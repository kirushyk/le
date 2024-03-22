/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#include <le/tensors/letensor.h>

#ifndef __LEPOLYNOMIA_H__
#define __LEPOLYNOMIA_H__

#include <stdio.h>
#include <glib.h>

G_BEGIN_DECLS

LeTensor * le_matrix_new_polynomia        (const LeTensor *matrix);

G_END_DECLS

#endif
