#include <cstdlib>
#include <cstring>
#include <string>
#include <le.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tensor.hpp"
#include "svm.hpp"

namespace py = pybind11;

static float logistic_loss(const le::Tensor &h, const le::Tensor &y)
{
    return le_logistic_loss(h.c_tensor(), y.c_tensor());
}

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
    py::class_<le::Model>(m, "Model")
        .def("predict", &le::Model::predict);
    py::class_<PySVM, le::Model>(m, "SVM")
        .def(py::init<>())
        .def("train", &PySVM::pyTrain, py::arg("x"), py::arg("y"), py::arg("kernel") = le::Kernel::LINEAR, py::arg("c") = 1.0f)
        .def("predict", &PySVM::predict);
    py::class_<PyLogisticClassifier, le::Model>(m, "LogisticClassifier")
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
    py::class_<le::Layer>(m, "Layer");
    py::class_<le::DenseLayer, le::Layer>(m, "DenseLayer")
        .def(py::init<std::string, unsigned, unsigned>());
    py::class_<le::ActivationLayer, le::Layer>(m, "ActivationLayer")
        .def(py::init<std::string, le::Activation>());
    py::class_<le::Sequential, le::Model>(m, "Sequential")
        .def(py::init<>())
        .def("add", &le::Sequential::add)
        .def("setLoss", &le::Sequential::setLoss)
        .def("predict", &le::Sequential::predict);
    py::class_<le::Optimizer>(m, "Optimizer")
        .def("step", &le::Optimizer::step);
    py::class_<le::BGD, le::Optimizer>(m, "BGD")
        .def(py::init<le::Model, le::Tensor, le::Tensor, float>(), py::arg("model"), py::arg("x"), py::arg("y"), py::arg("learning_rate") = 1.0f)
        .def("step", &le::BGD::step);
    m.doc() = "Le Python Binding";
    m.def("tensor", &tensor, "Create a Le Tensor");
    m.def("logistic_loss", &logistic_loss, "Compares output with ground truth");
}
