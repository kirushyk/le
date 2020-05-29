#include "bgd.hpp"

using namespace le;

BGD::BGD(Model model, Tensor input, Tensor output, float learning_rate)
{

}

BGD::~BGD()
{
    
}

void BGD::step()
{
    le_bgd_step(c_optimizer());
}

