/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LELOGISTIC_H__
#define __LELOGISTIC_H__

#include <le/lemacros.h>
#include <le/tensors/letensor.h>

LE_BEGIN_DECLS

typedef struct LeLogisticClassifier LeLogisticClassifier;

#define LE_LOGISTIC_CLASSIFIER(obj) ((LeLogisticClassifier *)(obj))

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
    float                    learning_rate;
    LeRegularization         regularization;
    float                    lambda;
    unsigned                 max_iterations;
} LeLogisticClassifierTrainingOptions;

void                    le_logistic_classifier_train       (LeLogisticClassifier *  classifier,
                                                            const LeTensor *        x_train,
                                                            const LeTensor *        y_train,
                                                            LeLogisticClassifierTrainingOptions
                                                                                    options);

void                    le_logistic_classifier_free        (LeLogisticClassifier *  classifier);

LE_END_DECLS

#endif
