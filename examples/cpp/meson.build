# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../..', '../../bindings/cpp')

executable('polynomial-logistic-regression-cpp', 'polynomial-logistic-regression.cpp',
    include_directories: incdir,
    link_with: [
        le,
        le_cpp
    ],
    install: false
)

executable('support-vector-machine-cpp', 'support-vector-machine.cpp',
    include_directories: incdir,
    link_with: [
        le,
        le_cpp
    ],
    install: false
)

executable('shallow-neural-network-cpp', 'shallow-neural-network.cpp',
    include_directories: incdir,
    link_with: [
        le,
        le_cpp
    ],
    install: false
)
