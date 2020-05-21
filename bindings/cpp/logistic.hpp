#pragma once
#include "model.hpp"
#include "tensor.hpp"

namespace le
{

enum class Regularization
{
    NONE = LE_REGULARIZATION_NONE,
    L1 = LE_REGULARIZATION_L1,
    L2 = LE_REGULARIZATION_L2
};

class LogisticClassifier: public Model
{
public:
    struct TrainingOptions
    {
        unsigned         polynomiaDegree;
        float            learningRate;
        Regularization   regularization;
        float            lambda;
        unsigned         maxIterations;
    };

    LogisticClassifier();
    
    void train(const Tensor &x_train, const Tensor &y_train, TrainingOptions options);

};

}
