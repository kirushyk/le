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
        unsigned         polynomia_degree;
        float            learning_rate;
        Regularization   regularization;
        float            lambda;
        unsigned         max_iterations;
    };

    void train(const Tensor &x_train, const Tensor &y_train, TrainingOptions options);

};

}
