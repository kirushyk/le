# Le - Machine Learning Framework.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platforms](https://img.shields.io/badge/platform-macos%20%7C%20linux-lightgrey.svg)
![Interfaces](https://img.shields.io/badge/interface-c%20%7C%20c%2B%2B%20%7C%20rust%20%7C%20python-blue.svg)

Le is a low-level machine learning library designed for readability and ease of use. Written in pure C, it, however, employs an object-oriented approach through GObject. Bindings are provided for other languages, allowing Le to be used in C++, Rust, and Python programs.

Le is currently under active development. New features and updates are on the way, so stay tuned!

At this moment following ML models are implemented:
* Polynomial Regression.
* Support Vector Machines (SVM).
* Sequential Feed-forward Neural Network (Multiple Layer Perceptron, MLP).
* k-Nearest Neighbors Algorithm (k-NN).

Optimization algorithms supported:
* Batch Gradient Descent (BGD).
* Stochastic Gradient Descent (SGD) with momentum.
* Sequential Minimal Optimization (SMO).

Supported backends:
* NVIDIA CUDA.
* Apple Metal.

## Installation

* [From Source](BUILD.md)
    
## Examples

* [C Examples](/examples/c)
* [C++ Examples](/examples/cpp)
* [Rust Examples](/examples/rust)
* [Python Examples](/examples/python)

## Tools

* [Le Board](https://github.com/kirushyk/le-board);
* [Le Playground](/tools/playground);
* [Le MNIST Inspect](/tools/mnist-inspect).

## License

Copyright &copy; 2017 Kyrylo Polezhaiev. All rights reserved.

Le is released under the [MIT](LICENSE) License.
