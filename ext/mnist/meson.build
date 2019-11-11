# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

mnist_sources = [
    'leidx.c',
    'lemnist.c'
]

mnist_deps = [
    cc.find_library('m'),
    cc.find_library('z'),
]

mnist = static_library('le_mnist', mnist_sources,
    dependencies: mnist_deps,
    include_directories: incdir,
    link_with: le,
)

mnist_path = get_option('datadir') / 'le' / 'ext' / 'mnist'

config = configuration_data()
config.set_quoted('MNIST_DATASET_PATH', get_option('prefix') / mnist_path)
configure_file(output: 'mnist-config.h', configuration: config)

install_data(
    [
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz'
    ],
    install_dir : mnist_path
)
