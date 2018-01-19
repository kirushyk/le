#include "letrainingdata.h"
#include <stdlib.h>

struct LeTrainingData
{
    LeMatrix *x;
    LeMatrix *y;
};

LeTrainingData *
le_training_data_new_copy (LeMatrix *x, LeMatrix *y)
{
    LeTrainingData *data = malloc(sizeof(LeTrainingData));
    data->x = le_matrix_new_copy(x);
    data->y = le_matrix_new_copy(y);
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

LeMatrix *
le_training_data_get_input(LeTrainingData *data)
{
    return data->x;
}

LeMatrix *
le_training_data_get_output (LeTrainingData *data)
{
    return data->y;
}

void
le_training_data_free(LeTrainingData *self)
{
    free(self);
}
