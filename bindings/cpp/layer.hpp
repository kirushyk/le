#pragma once
#include <memory>

namespace le
{

class Layer
{
public:
    Layer();
    ~Layer();

private:
    struct Private;
    std::shared_ptr<Private> priv;

};

}
