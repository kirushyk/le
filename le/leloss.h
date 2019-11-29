/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LELOSS_H__
#define __LELOSS_H__

#include "lematrix.h"

float le_cross_entropy             (LeTensor *predictions,
                                    LeTensor *labels);

float le_one_hot_misclassification (LeTensor *predictions,
                                    LeTensor *labels);

#endif
