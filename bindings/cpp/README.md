## C++ Bindings for Le

This is C++ interface for Le machine learning framework.

[Tensor](tensor.hpp)
* Scalar - Rank 0 Tensor with single element
* Vector - Rank 1 Tensor
* Matrix - Rank 2 Tensor

[Optimizer](optimizer.hpp) - Base Class for Optimizers
* [BGD](bgd.hpp) - Vanilla Gradient Descent
* SGD - Stochastic Gradient Descent with momentum

[Model](model.hpp) - Base Class for ML Models
* PolynomialClassifier - Logistic Regression with Polynomial features
* [SVM](svm.hpp) - Support Vector Machine
* [Sequential](sequential.hpp) - Multiple Layer Perceptron

[Layer](layer.hpp)
* [Dense](dense-layer.hpp)
* Activation
* Conv2D
