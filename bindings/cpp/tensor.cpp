#include "le.hpp"
#include <cstdint>
#include <cstdarg>
#include <iomanip>
#include <le/le.h>
#include <le/tensors/letensor-imp.h>

using namespace le;

struct Tensor::Private
{
    LeTensor *tensor;
};

Tensor::Tensor(Type t, unsigned num_dimensions, ...):
    priv(std::make_shared<Private>())
{
    std::va_list args;
    va_start(args, num_dimensions);
    priv->tensor = le_tensor_new_from_va_list((LeType)t, num_dimensions, args);
    va_end(args);
}

Tensor::Tensor(Type t, Shape s):
    priv(std::make_shared<Private>())
{
    priv->tensor = le_tensor_new_uninitialized((LeType)t, le_shape_copy(s.c_shape()));
}

Tensor::Tensor(const Tensor &tensor):
    priv(std::make_shared<Private>())
{
    priv->tensor = le_tensor_new_copy(tensor.c_tensor());
}

Tensor::Tensor(Tensor &&tensor):
    priv(std::make_shared<Private>())
{
    priv->tensor = tensor.priv->tensor;
    tensor.priv->tensor = NULL;
}

Tensor::Tensor(LeTensor *c_tensor):
    priv(std::make_shared<Private>())
{
    priv->tensor = c_tensor;
}

Tensor::~Tensor()
{
    le_tensor_unref(priv->tensor);
}

LeTensor *Tensor::c_tensor() const
{
    return priv->tensor;
}

void * Tensor::data() const
{
    return priv->tensor->data;
}

#define TENSOR_PRINT_MAX_SIZE 10

std::ostream & le::operator << (std::ostream &output, const Tensor &tensor)
{
    LeTensor *c_tensor = tensor.priv->tensor;
    if (c_tensor->shape->num_dimensions != 2)
    {
        output << "<" << c_tensor->shape->num_dimensions << "D tensor>" << std::endl;
    }

    output << '[';
    for (std::uint32_t y = 0; (y < c_tensor->shape->sizes[0]) && (y < TENSOR_PRINT_MAX_SIZE); y++)
    {
        std::uint32_t x;
        for (x = 0; (x < c_tensor->shape->sizes[1]) && (x < TENSOR_PRINT_MAX_SIZE); x++)
        {
            switch (tensor.priv->tensor->element_type)
            {
            case LE_TYPE_U8:
                output << (unsigned)((std::uint8_t *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                break;
            case LE_TYPE_I8:
                output << (int)((std::int8_t *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                break;
            case LE_TYPE_I16:
                output << (int)((std::int16_t *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                break;
            case LE_TYPE_I32:
                output << (int)((std::int32_t *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                break;
            case LE_TYPE_F32:
                output << std::fixed << std::setprecision(3) << 
                    ((float *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                break;
            case LE_TYPE_F64:
                output << std::fixed << std::setprecision(3) <<
                    ((double *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                break;
            case LE_TYPE_VOID:
            default:
                output << '?';
                break;
            }
            if (x < c_tensor->shape->sizes[1] - 1)
            {
                output << ' ';
            }
        }
        if (x < c_tensor->shape->sizes[1])
        {
            output << "...";
        }
        if (y < c_tensor->shape->sizes[0] - 1)
        {
            output << ';' << std::endl << ' ';
        }
    }
    output << ']';

    return output;
}
