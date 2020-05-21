#include "activation-layer.hpp"

using namespace le;

ActivationLayer::ActivationLayer(std::string name, Activation activation)
{
    LeActivationLayer *c_layer = le_activation_layer_new(name.c_str(), (LeActivation)activation);
    setCLayer(LE_LAYER(c_layer));
}
