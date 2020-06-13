# Build Le from Source

## Dependencies

You will need C11 compiler. If you want Python bindings you should have C++14 compiler and [Pybind11](https://github.com/pybind/pybind11) installed. 

### Ubuntu

Use APT package manager to install dependencies:

    sudo apt install meson gcc g++ zlib1g-dev libpython3-dev python3-pybind11 libgtk-3-dev

### macOS

Use [Homebrew](https://brew.sh) to install packages needed:

    brew install meson pybind11 gtk+3

## Configuration

[Meson Build system](https://mesonbuild.com) is used to automate building and installation. To create build directory, type:

    meson {source_directory_name} {build_directory_name}

You can put your build directory inside of source directory:

    cd {source_directory_name}
    meson {build_directory_name}

## Compilation

Use [`ninja`](https://ninja-build.org) command from your build directory to compile binaries:

    cd {build_directory_name}
    ninja

## Installation

To install compiled binaries locally, type:

    sudo ninja install
