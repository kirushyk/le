#pragma once
#include <string>
#include "layer.hpp"

namespace le
{

class DenseLayer: public Layer
{
public:
    DenseLayer(std::string name, unsigned inputs, unsigned outputs);
    
};

}
