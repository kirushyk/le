#pragma once
#include "model.hpp"
#include "tensor.hpp"
#include "layer.hpp"

namespace le
{

enum class Loss
{
    MSE = LE_LOSS_MSE,
    LOGISTIC = LE_LOSS_LOGISTIC,
    CROSS_ENTROPY = LE_LOSS_CROSS_ENTROPY
};

class Sequential: public Model
{
public:
    Sequential();
    void add(Layer layer);
    void setLoss(Loss);
    
};

}
