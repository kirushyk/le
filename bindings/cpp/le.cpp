#include "le.hpp"
#include <le/le.h>

using namespace le;

struct Tensor::Private
{
    LeTensor *tensor;
};

Tensor::Tensor():
    priv(std::make_shared<Private>())
{
    priv->tensor = le_tensor_new(LE_TYPE_VOID, 0, NULL);
}

Tensor::~Tensor()
{
    le_tensor_free(priv->tensor);
}
