#pragma once
#include "model.hpp"
#include "tensor.hpp"

namespace le
{

enum class Kernel
{
    LINEAR = LE_KERNEL_LINEAR,
    RBF = LE_KERNEL_RBF
};

class SVM: public Model
{
public:
    struct TrainingOptions
    {
        Kernel kernel;
        float  c;
    };

    SVM();
    
    void train(const Tensor &x_train, const Tensor &y_train, TrainingOptions options);

};

}
