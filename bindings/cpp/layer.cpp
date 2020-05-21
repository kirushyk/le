#include "layer.hpp"
#include <le/le.h>

using namespace le;

struct Layer::Private
{

};

Layer::Layer():
    priv(std::make_shared<Private>())
{

}

Layer::~Layer()
{

}
