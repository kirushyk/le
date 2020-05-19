#include <pybind11/pybind11.h>

namespace py = pybind11;

int
tensor()
{
    return 0;
}

PYBIND11_MODULE(le, m)
{
    m.doc() = "Le Python Binding";
    m.def("tensor", &tensor, "Create a Le Tensor");
}
