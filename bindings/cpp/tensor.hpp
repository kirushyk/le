#pragma once
#include <memory>
#include <iostream>
#include "type.hpp"

namespace le
{

class Tensor
{
public:
    Tensor(Type t, unsigned num_dimensions, ...);
    Tensor(const Tensor &tensor);
    Tensor(LeTensor *c_tensor);
    ~Tensor();

    const LeTensor *c_tensor() const;

    friend std::ostream & operator << (std::ostream &out, const Tensor &tensor);

private:
    struct Private;
    std::shared_ptr<Private> priv;

};

}