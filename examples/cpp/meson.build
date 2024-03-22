# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

inc_cpp = include_directories('../../bindings/cpp')

if have_cpp
    executable('polynomial-logistic-regression-cpp', 'polynomial-logistic-regression.cpp',
        include_directories: [
            inc,
            inc_cpp
        ],
        link_with: [
            le_cpp
        ],
        install: false,
        dependencies: glib
    )

    executable('support-vector-machine-cpp', 'support-vector-machine.cpp',
        include_directories: [
            inc,
            inc_cpp
        ],
        link_with: [
            le_cpp
        ],
        install: false,
        dependencies: glib
    )

    executable('shallow-neural-network-cpp', 'shallow-neural-network.cpp',
        include_directories: [
            inc,
            inc_cpp
        ],
        link_with: [
            le_cpp
        ],
        install: false,
        dependencies: glib
    )
endif
