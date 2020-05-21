#include "sequential.hpp"
#include <le/le.h>

using namespace le;

Sequential::Sequential()
{
    setCModel(LE_MODEL(le_sequential_new()));
}

void Sequential::add(Layer layer)
{
    le_sequential_add((LeSequential *)c_model(), layer.c_layer());
}

void Sequential::setLoss(Loss loss)
{
    le_sequential_set_loss((LeSequential *)c_model(), (LeLoss)loss);
}
