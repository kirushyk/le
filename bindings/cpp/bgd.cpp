#include "bgd.hpp"

using namespace le;

BGD::BGD(Model model, std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output, float learning_rate)
{
    LeOptimizer *c_optimizer = LE_OPTIMIZER(le_bgd_new(model.c_model(), input->c_tensor(), output->c_tensor(), learning_rate));
    setCOptimizer(c_optimizer);
}

BGD::~BGD()
{

}

void BGD::step()
{
    le_bgd_step(c_optimizer());
}

