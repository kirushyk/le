/**
 *
 */

#ifndef __LELOGISTIC_H_
#define __LELOGISTIC_H_

#include "lematrix.h"

typedef struct LeLogisticClassifier LeLogisticClassifier;

LeLogisticClassifier * le_logistic_classifier_new_train (LeMatrix             *x_train,
                                                         LeMatrix             *y_train);

LeMatrix *             le_logistic_classifier_prefict   (LeLogisticClassifier *classifier,
                                                         LeMatrix             *x);

void                   le_logistic_classifier_free      (LeLogisticClassifier *classifier);

#endif
