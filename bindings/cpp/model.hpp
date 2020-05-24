#pragma once
#include <memory>
#include <le/models/lemodel.h>
#include "tensor.hpp"

namespace le
{
    
class Model
{
public:
    Model();
    virtual ~Model();

    LeModel *c_model();

    Tensor predict(Tensor input);

protected:
    void setCModel(LeModel *c_model);

private:
    struct Private;
    std::shared_ptr<Private> priv;

};

}
