/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LELOGISTIC_H__
#define __LELOGISTIC_H__

#include <glib.h>
#include "lemodel.h"
#include <le/tensors/letensor.h>

G_BEGIN_DECLS

G_DECLARE_FINAL_TYPE (LeLogisticClassifier, le_logistic_classifier, LE, LOGISTIC_CLASSIFIER, LeModel);

// typedef struct LeLogisticClassifier LeLogisticClassifier;

// #define LE_LOGISTIC_CLASSIFIER(obj) ((LeLogisticClassifier *)(obj))

LeLogisticClassifier * le_logistic_classifier_new     (void);

typedef enum LeRegularization
{
    LE_REGULARIZATION_NONE,
    LE_REGULARIZATION_L1,
    LE_REGULARIZATION_L2
} LeRegularization;

typedef struct LeLogisticClassifierTrainingOptions
{
    unsigned                 polynomia_degree;
    gfloat                    learning_rate;
    LeRegularization         regularization;
    gfloat                    lambda;
    unsigned                 max_iterations;
} LeLogisticClassifierTrainingOptions;

void                    le_logistic_classifier_train       (LeLogisticClassifier *  classifier,
                                                            const LeTensor *        x_train,
                                                            const LeTensor *        y_train,
                                                            LeLogisticClassifierTrainingOptions
                                                                                    options);

G_END_DECLS

#endif
