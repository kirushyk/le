#include "tensor.hpp"
#include <le.hpp>

py::object tensor(py::array_t<float> elements)
{
    auto r = elements.unchecked<>();
    le::Shape s(elements.ndim());
    for (int i = 0; i < elements.ndim(); i++) 
    {
        s[i] = r.shape(i);
    }
    unsigned numElements = le_shape_get_elements_count(s.c_shape());
    le::Tensor t(le::Type::F32, s);
    std::memcpy(t.data(), r.data(0), numElements * sizeof(float));
    return py::cast(t);
}
