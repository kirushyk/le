#include "layer.hpp"

using namespace le;

struct Layer::Private
{
    LeLayer *c_layer;
};

Layer::Layer():
    priv(std::make_shared<Private>())
{
    priv->c_layer = NULL;
}

Layer::~Layer()
{

}

void Layer::setCLayer(LeLayer *c_layer)
{
    priv->c_layer = c_layer;
}

LeLayer * Layer::c_layer()
{
    return priv->c_layer;
}

