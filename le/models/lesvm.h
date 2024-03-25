/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LESVM_H__
#define __LESVM_H__

#include <glib.h>
#include "lemodel.h"
#include <le/tensors/letensor.h>

G_BEGIN_DECLS

typedef enum LeKernel
{
    LE_KERNEL_LINEAR,
    LE_KERNEL_RBF
} LeKernel;

// typedef struct LeSVM LeSVM;

// #define LE_SVM(obj) ((LeSVM *)(obj))

G_DECLARE_FINAL_TYPE (LeSVM, le_svm, LE, SVM, LeModel);

LeSVM *                 le_svm_new                         (void);

typedef struct LeSVMTrainingOptions
{
    LeKernel kernel;
    float    c;
} LeSVMTrainingOptions;

void                    le_svm_train                       (LeSVM *                 svm,
                                                            const LeTensor *        x_train,
                                                            const LeTensor *        y_train,
                                                            LeSVMTrainingOptions    options);

G_END_DECLS

#endif
