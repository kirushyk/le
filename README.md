# Le - Machine Learning Framework.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platforms](https://img.shields.io/badge/platform-macos%20%7C%20linux-lightgrey.svg)
![Interfaces](https://img.shields.io/badge/interface-c%20%7C%20c%2B%2B%20%7C%20rust%20%7C%20python-blue.svg)

Le is Machine Learning Framework designed so that programs using it will be easy to read. Library is written in pure C but in object-oriented way. Bindings to other languages are provided so Le can be used by C++ and Python programs. Rust and Node.js interface will be added later.

Le is now under heavy development. Please come back soon.

At this moment following ML models are implemented:
* Polynomial Regression.
* Support Vector Machines (SVM).
* Sequential Feed-forward Neural Network (Multiple Layer Perceptron, MLP).

Optimization algorithms supported:
* Batch Gradient Descent (BGD).
* Sequential Minimal Optimization (SMO).
* Stochastic Gradient Descent (SGD) with momentum.

## Installation

### From Source

You will need C11 compiler. If you want Python bindings you should have C++14 compiler and [Pybind11](https://github.com/pybind/pybind11) installed. [Meson Build system](https://mesonbuild.com) is used to automate building and installation. To create build directory, type:

    meson {source_directory_name} {build_directory_name}

You can put your build directory inside of source directory:

    cd {source_directory_name}
    meson {build_directory_name}

Then use [`ninja`](https://ninja-build.org) command from your build directory to compile binaries:

    cd {build_directory_name}
    ninja

To install compiled binaries locally, type:

    sudo ninja install
    
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

Copyright &copy; 2017 Kyrylo Polezhaiev and contributors. All rights reserved.

Le is released under the [MIT](LICENSE) License.
