#pragma once
#include "model.hpp"
#include "tensor.hpp"
#include "layer.hpp"

namespace le
{

class Sequential: public Model
{
public:
    Sequential();
    void add(Layer layer);
    
};

}
