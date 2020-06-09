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
    void pyTrain(const le::Tensor &x_train, const le::Tensor &y_train, const le::Kernel kernel, const float c)
    {
        le::SVM::TrainingOptions options;
        options.kernel = le::Kernel::LINEAR;
        options.c = 1.0f;
        train(x_train, y_train, options);
    }
};

class PyLogisticClassifier: public le::LogisticClassifier
{
public: 
    void pyTrain(const le::Tensor &x_train, const le::Tensor &y_train)
    {
        le::LogisticClassifier::TrainingOptions options;
        options.maxIterations = 100;
        options.learningRate = 1.0f;
        options.polynomiaDegree = 1;
        options.regularization = le::Regularization::NONE;
        options.lambda = 0.0f;
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
    py::enum_<le::Kernel>(m, "Kernel")
        .value("LINEAR", le::Kernel::LINEAR)
        .value("RBF", le::Kernel::RBF);
    py::class_<PySVM>(m, "SVM")
        .def(py::init<>())
        .def("train", &PySVM::pyTrain, py::arg("x"), py::arg("y"), py::arg("kernel") = le::Kernel::LINEAR, py::arg("c") = 1.0f)
        .def("predict", &PySVM::predict);
    py::class_<PyLogisticClassifier>(m, "LogisticClassifier")
        .def(py::init<>())
        .def("train", &PyLogisticClassifier::pyTrain)
        .def("predict", &PyLogisticClassifier::predict);
    py::enum_<le::Loss>(m, "Loss")
        .value("MSE", le::Loss::MSE)
        .value("LOGISTIC", le::Loss::LOGISTIC)
        .value("CROSS_ENTROPY", le::Loss::CROSS_ENTROPY);
    py::enum_<le::Activation>(m, "Activation")
        .value("LINEAR", le::Activation::LINEAR)
        .value("SIGMOID", le::Activation::SIGMOID)
        .value("TANH", le::Activation::TANH)
        .value("RELU", le::Activation::RELU)
        .value("SOFTMAX", le::Activation::SOFTMAX);
    py::class_<le::Sequential>(m, "Sequential")
        .def(py::init<>())
        .def("predict", &le::Sequential::predict);
    m.doc() = "Le Python Binding";
    m.def("tensor", &tensor, "Create a Le Tensor");
}
