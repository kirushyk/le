#include "dense-layer.hpp"
#include <le/models/layers/ledenselayer.h>

using namespace le;

DenseLayer::DenseLayer(std::string name, unsigned inputs, unsigned outputs)
{
    LeDenseLayer *c_layer = le_dense_layer_new(name.c_str(), inputs, outputs);
    setCLayer(LE_LAYER(c_layer));
}
