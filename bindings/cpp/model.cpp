#include "model.hpp"

using namespace le;

struct Model::Private
{
    LeModel *model;
};

Model::Model():
    priv(std::make_shared<Private>())
{
    priv->model = NULL;
}

Model::~Model()
{
    
}

LeModel * Model::c_model()
{
    return priv->model;
}

void Model::setCModel(LeModel *c_model)
{
    priv->model = c_model;
}

Tensor Model::predict(Tensor input)
{
    return le_model_predict(c_model(), input.c_tensor());
}
