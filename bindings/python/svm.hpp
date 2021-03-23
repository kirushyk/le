#pragma once
#include <le.hpp>

class PySVM: public le::SVM
{
public: 
    void pyTrain(const le::Tensor &x_train, const le::Tensor &y_train, const le::Kernel kernel, const float c)
    {
        le::SVM::TrainingOptions options;
        options.kernel = le::Kernel::LINEAR;
        options.c = 1.0f;
        train(x_train, y_train, options);
    }
};
