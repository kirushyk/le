#pragma once
#include <string>
#include "layer.hpp"
#include <le/models/layers/leactivationlayer.h>

namespace le
{

enum class Activation
{
    LINEAR = LE_ACTIVATION_LINEAR,
    SIGMOID = LE_ACTIVATION_SIGMOID,
    TANH = LE_ACTIVATION_TANH,
    RELU = LE_ACTIVATION_RELU,
    SOFTMAX = LE_ACTIVATION_SOFTMAX
};

class ActivationLayer: public Layer
{
public:
    ActivationLayer(std::string name, Activation activation);

};

}
