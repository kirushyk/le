/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LELOGISTIC_H__
#define __LELOGISTIC_H__

#include "lematrix.h"

typedef struct LeLogisticClassifier LeLogisticClassifier;

LeLogisticClassifier * le_logistic_classifier_new     (void);

typedef enum LeRegularization
{
    LE_REGULARIZATION_NONE,
    LE_REGULARIZATION_L1,
    LE_REGULARIZATION_L2
} LeRegularization;

typedef struct LeLogisticClassifierTrainingOptions
{
    unsigned         polynomia_degree;
    float            alpha;
    LeRegularization regularization;
    float            lambda;
} LeLogisticClassifierTrainingOptions;

void                   le_logistic_classifier_train   (LeLogisticClassifier *classifier,
                                                       LeMatrix             *x_train,
                                                       LeMatrix             *y_train,
                                                       LeLogisticClassifierTrainingOptions
                                                                             options);

void                   le_logistic_classifier_free    (LeLogisticClassifier *classifier);

#endif
