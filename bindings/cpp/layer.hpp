#pragma once
#include <memory>
#include <le/models/layers/lelayer.h>

namespace le
{

class Layer
{
public:
    Layer();
    ~Layer();

    LeLayer *c_layer();

protected:
    void setCLayer(LeLayer *c_layer);

private:
    struct Private;
    std::shared_ptr<Private> priv;

};

}
