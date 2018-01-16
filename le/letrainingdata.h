#ifndef __LETRAININGDATA_H__
#define __LETRAININGDATA_H__

#include "lematrix.h"

typedef struct LeTrainingData LeTrainingData;

LeTrainingData * le_training_data_new_copy (LeMatrix *input,
                                            LeMatrix *output);

LeTrainingData * le_training_data_new_take (LeMatrix *input,
                                            LeMatrix *output);

void             le_training_data_free     (LeTrainingData *);

#endif
