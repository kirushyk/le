#pragma once
#include "optimizer.hpp"
#include "model.hpp"
#include "tensor.hpp"

namespace le
{

class BGD: public Optimizer
{
public:
    BGD(Model model, Tensor input, Tensor output, float learning_rate);
    virtual ~BGD();

    virtual void step() override;

};

}
