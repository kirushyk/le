#include <cstdlib>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <le.hpp>

namespace py = pybind11;

py::object tensor(py::array_t<double> elements)
{
    auto r = elements.unchecked<>();
    unsigned *shapeData = (unsigned *)std::malloc(elements.ndim() * sizeof(unsigned));
    for (int i = 0; i < elements.ndim(); i++) 
    {
        shapeData[i] = r.shape(i);
    }
    le::Shape s(elements.ndim(), shapeData);
    float *tensorData = (float *)std::malloc(le_shape_get_elements_count(s.c_shape()) * sizeof(float));
    le::Tensor t(le::Type::FLOAT32, s, tensorData);
    // double sum = 0;
    // for (ssize_t i = 0; i < r.shape(0); i++)
    //     for (ssize_t j = 0; j < r.shape(1); j++)
    //         for (ssize_t k = 0; k < r.shape(2); k++)
    //             sum += r(i, j, k);
    // return sum;
    return py::cast(t);
}

PYBIND11_MODULE(le, m)
{
    py::class_<le::Tensor>(m, "Tensor")
        .def("__str__", [](const le::Tensor &tensor)
        {
            return le_tensor_to_cstr(tensor.c_tensor());
        });
    m.doc() = "Le Python Binding";
    m.def("tensor", &tensor, "Create a Le Tensor");
}
