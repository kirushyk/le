#ifndef __LESVM_H__
#define __LESVM_H__

#include "lematrix.h"

typedef struct LeSVM LeSVM;

LeSVM *                le_svm_new_train                 (LeMatrix             *x_train,
                                                         LeMatrix             *y_train);

LeMatrix *             le_svm_predict                   (LeSVM                *svm,
                                                         LeMatrix             *x);

void                   le_svm_free                      (LeSVM                *svm);

#endif
