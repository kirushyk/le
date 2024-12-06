#include "model.hpp"

using namespace le;

struct Shape::Private
{
    LeShape *c_shape;
};

Shape::Shape(unsigned numDimensions):
    priv(std::make_shared<Private>())
{
    priv->c_shape = le_shape_new_uninitialized(numDimensions);
}

Shape::Shape(const Shape &another):
    priv(std::make_shared<Private>())
{
    priv->c_shape = le_shape_copy(another.priv->c_shape);
}

LeShape *Shape::c_shape()
{
    return priv->c_shape;
}

std::uint32_t & Shape::operator [](std::size_t index)
{
    return priv->c_shape->sizes[index];
}

Shape::~Shape()
{
    le_shape_free(priv->c_shape);
}
