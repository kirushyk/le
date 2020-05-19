#include <pybind11/pybind11.h>

namespace py = pybind11;

int
le_tensor()
{
    return 0;
}

PYBIND11_MODULE(example, m)
{
    m.doc() = "Le Python Binding";
    m.def("tensor", &le_tensor, "Create a Le Tensor");
}
