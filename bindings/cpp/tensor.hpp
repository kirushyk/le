#pragma once
#include <memory>
#include <iostream>
#include "type.hpp"
#include "shape.hpp"

namespace le
{

class Tensor
{
public:
    Tensor(Type t, unsigned num_dimensions, ...);
    Tensor(Type t, Shape s);
    Tensor(const Tensor &tensor);
    Tensor(Tensor &&tensor);
    Tensor(LeTensor *c_tensor);
    ~Tensor();

    LeTensor * c_tensor() const;
    void * data() const;

    friend std::ostream & operator << (std::ostream &out, const Tensor &tensor);

private:
    struct Private;
    std::shared_ptr<Private> priv;

};

std::ostream & operator << (std::ostream &out, const Tensor &tensor);

}