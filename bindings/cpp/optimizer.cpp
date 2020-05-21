#include "optimizer.hpp"

using namespace le;

struct Optimizer::Private
{
    LeOptimizer *c_optimizer;
};

Optimizer::Optimizer():
    priv(std::make_shared<Private>())
{
    priv->c_optimizer = nullptr;
}

Optimizer::~Optimizer()
{

}

LeOptimizer * Optimizer::c_optimizer()
{
    return priv->c_optimizer;
}

void Optimizer::setCOptimizer(LeOptimizer *c_optimizer)
{
    priv->c_optimizer = c_optimizer;
}
