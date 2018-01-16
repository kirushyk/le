#include "letrainingdata.h"
#include <stdlib.h>

struct LeTrainingData
{
    LeMatrix *x;
    LeMatrix *y;
};

LeTrainingData *
le_training_data_new_copy (LeMatrix *input, LeMatrix *output)
{
    LeTrainingData *data = malloc(sizeof(LeTrainingData));
    data->x = le_matrix_new_copy(input);
    data->y = le_matrix_new_copy(output);
    return data;
}

LeTrainingData *
le_training_data_new_take(LeMatrix *input, LeMatrix *output)
{
    LeTrainingData *data = malloc(sizeof(LeTrainingData));
    data->x = input;
    data->y = output;
    return data;
}

void
le_training_data_free(LeTrainingData *self)
{
    free(self);
}
