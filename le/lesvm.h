/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LESVM_H__
#define __LESVM_H__

#include "letensor.h"

typedef enum LeKernel
{
    LE_KERNEL_LINEAR,
    LE_KERNEL_RBF
} LeKernel;

typedef struct LeSVM LeSVM;

LeSVM *                le_svm_new                       (void);

typedef struct LeSVMTrainingOptions
{
    LeKernel kernel;
    float    c;
} LeSVMTrainingOptions;

void                   le_svm_train                     (LeSVM                *svm,
                                                         LeTensor             *x_train,
                                                         LeTensor             *y_train,
                                                         LeSVMTrainingOptions  options);

void                   le_svm_free                      (LeSVM                *svm);

#endif
