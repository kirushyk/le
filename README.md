# Le - Machine Learning Framework.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platforms](https://img.shields.io/badge/platform-macos%20%7C%20linux-lightgrey.svg)
![Interfaces](https://img.shields.io/badge/interface-c%20%7C%20c%2B%2B%20%7C%20rust%20%7C%20python-blue.svg)

**Le** is a low-level machine learning library designed for readability and ease of use. Written in pure C, it, however, employs an object-oriented approach through GObject.

Bindings are available for other languages, allowing **Le** to be used in C++ and Python programs, with Rust support planned for the future.

I originally created **Le** as a learning project while studying Andrew Ng's Deep Learning course. At the time, I didn’t feel confident or fluent in Python, so I chose to implement everything in C to gain a deeper understanding and maintain control. Today, I use it to re-implement or experiment with models from research papers or open-source releases.

The name **Le** is short for "Learning". **Le** is currently under active development, with new features and updates on the way. Stay tuned!

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

* [From Source](/docs/BUILD.md)
    
## Examples

* [C Examples](/examples/c)
* [C++ Examples](/examples/cpp)
* [Rust Examples](/examples/rust)
* [Python Examples](/examples/python)

## Contribution

* [Code Style Guidelines](/docs/HACKING.md)

## Tools

* [Le Board](https://github.com/kirushyk/le-board);
* [Le Playground](/tools/playground);
* [Le MNIST Inspect](/tools/mnist-inspect).

## License

Copyright &copy; 2017 Kyrylo Polezhaiev. All rights reserved.

Le is released under the [MIT](LICENSE) License.
