#pragma once
#include <memory>
#include <le/optimization/leoptimizer.h>

namespace le
{

class Optimizer
{
public:
    Optimizer();
    virtual ~Optimizer();

    LeOptimizer *c_optimizer();
    virtual void step();

protected:
    void setCOptimizer(LeOptimizer *c_optimizer);

private:
    struct Private;
    std::shared_ptr<Private> priv;

};

}
