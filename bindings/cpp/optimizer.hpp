#pragma once
#include <memory>
#include <le/optimizers/leoptimizer.h>

namespace le
{

class Optimizer
{
public:
    Optimizer();
    ~Optimizer();

    LeOptimizer *c_optimizer();

protected:
    void setCOptimizer(LeOptimizer *c_optimizer);

private:
    struct Private;
    std::shared_ptr<Private> priv;

};

}
