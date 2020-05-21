#include "model.hpp"

using namespace le;

struct Shape::Private
{
    LeShape *c_shape;
};

Shape::Shape(unsigned numDimensions, std::uint32_t *sizes):
    priv(std::make_shared<Private>())
{
    priv->c_shape = le_shape_new_from_data(numDimensions, sizes);
}

LeShape *Shape::c_shape()
{
    return priv->c_shape;
}

Shape::~Shape()
{
    // le_shape_free(priv->c_shape);
}
