/* Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
   Released under the MIT license. See LICENSE file in the project root for full license information. */

#ifndef __LELOGISTIC_H_
#define __LELOGISTIC_H_

#include "lematrix.h"

typedef struct LeLogisticClassifier LeLogisticClassifier;

LeLogisticClassifier * le_logistic_classifier_new_train (LeMatrix             *x_train,
                                                         LeMatrix             *y_train,
                                                         unsigned              polynomia_degree);

LeMatrix *             le_logistic_classifier_predict   (LeLogisticClassifier *classifier,
                                                         LeMatrix             *x);

void                   le_logistic_classifier_free      (LeLogisticClassifier *classifier);

#endif
