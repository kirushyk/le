#include <cstdlib>
#include <cstring>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <le.hpp>

namespace py = pybind11;

py::object tensor(py::array_t<float> elements)
{
    auto r = elements.unchecked<>();
    unsigned *shapeData = (unsigned *)std::malloc(elements.ndim() * sizeof(unsigned));
    for (int i = 0; i < elements.ndim(); i++) 
    {
        shapeData[i] = r.shape(i);
    }
    le::Shape s(elements.ndim(), shapeData);
    unsigned numElements = le_shape_get_elements_count(s.c_shape());
    float *tensorData = (float *)std::malloc(numElements * sizeof(float));
    std::memcpy(tensorData, r.data(0), numElements * sizeof(float));
    le::Tensor t(le::Type::FLOAT32, s, tensorData);
    return py::cast(t);
}

class PySVM: public le::SVM
{
public: 
    void pyTrain(const le::Tensor &x_train, const le::Tensor &y_train)
    {
        le::SVM::TrainingOptions options;
        options.kernel = le::Kernel::LINEAR;
        options.c = 1.0f;
        train(x_train, y_train, options);
    }
};

PYBIND11_MODULE(le, m)
{
    py::class_<le::Tensor>(m, "Tensor")
        .def("__str__", [](const le::Tensor &tensor)
        {
            return le_tensor_to_cstr(tensor.c_tensor());
        });
    py::class_<PySVM>(m, "SVM")
        .def(py::init<>())
        .def("train", &PySVM::pyTrain)
        .def("predict", &PySVM::predict);
    m.doc() = "Le Python Binding";
    m.def("tensor", &tensor, "Create a Le Tensor");
}
