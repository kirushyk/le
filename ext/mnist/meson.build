# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

mnist_sources = [
    'leidx.c',
    'lemnist.c'
]

mnist_deps = [
    cc.find_library('m')
]

mnist = static_library('le_mnist', mnist_sources,
    dependencies: mnist_deps,
    include_directories: incdir,
    link_with: le,
)

