#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <le.hpp>
#include <string>

namespace py = pybind11;

py::object tensor(py::array_t<double> elements)
{
    le::Tensor t(le::Type::FLOAT32, 1, 1, 1.0);
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
