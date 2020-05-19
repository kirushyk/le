#include "le.hpp"
#include <cstdint>
#include <cstdarg>
#include <le/le.h>
#include <le/letensor-imp.h>

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

Tensor::~Tensor()
{
    le_tensor_free(priv->tensor);
}

#define TENSOR_PRINT_MAX_SIZE 10

std::ostream & le::operator << (std::ostream &output, const Tensor &tensor)
{
    LeTensor *c_tensor = tensor.priv->tensor;
    if (c_tensor->shape->num_dimensions != 2)
    {
        output << "<" << c_tensor->shape->num_dimensions << "dD tensor>" << std::endl;
    }

    for (int y = 0; (y < c_tensor->shape->sizes[0]) && (y < TENSOR_PRINT_MAX_SIZE); y++)
    {
        int x;
        for (x = 0; (x < c_tensor->shape->sizes[1]) && (x < TENSOR_PRINT_MAX_SIZE); x++)
        {
            switch (tensor.priv->tensor->element_type)
            {
                case LE_TYPE_UINT8:
                    std::cerr << (unsigned)((std::uint8_t *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                    break;
                case LE_TYPE_INT8:
                    std::cerr << (int)((std::int8_t *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                    break;
                case LE_TYPE_INT16:
                    std::cerr << (int)((std::int16_t *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                    break;
                case LE_TYPE_INT32:
                    std::cerr << (int)((std::int32_t *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                    break;
                case LE_TYPE_FLOAT32:
                    std::cerr << ((float *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                    break;
                case LE_TYPE_FLOAT64:
                    std::cerr << ((float *)c_tensor->data)[y * c_tensor->shape->sizes[1] + x];
                    break;
                case LE_TYPE_VOID:
                default:
                    std::cerr << '?';
                    break;
            }
            if (x < c_tensor->shape->sizes[1] - 1)
            {
                std::cerr << ' ';
            }
        }
        if (x < c_tensor->shape->sizes[1])
        {
            std::cerr << "...";
        }
        if (y < c_tensor->shape->sizes[0] - 1)
        {
            std::cerr << ';' << std::endl << ' ';
        }
    }

    return output;
}
