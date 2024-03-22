# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

config = configuration_data()
config.set_quoted('TEST_DIR', meson.current_build_dir())
configure_file(output: 'test-config.h', configuration: config)

le_tests = [
    ['list.c'],
    ['matrices.c'],
    ['tensor-view.c'],
    ['type-generic.c'],
    ['sobel.c'],
    ['relu.c'],
    ['tensorlist.c'],
    ['subtensor.c'],
    ['input_normalization.c'],
    # ['cnn-inf.c'],
    ['gradcheck.c']
]

le_tests_link_with = [
    le,
    tensorlist
]

if have_metal
    le_tests += [
        ['metal/hadamard.c'],
        ['metal/sigmoid.c'],
        ['metal/matmul.c']
    ]
    le_tests_link_with += le_metal
endif

if have_cuda
    le_tests += [
        ['cuda/hadamard.c'],
        ['cuda/sigmoid.c'],
        ['cuda/matmul.c']
    ]
    le_tests_link_with += le_cuda
endif

if have_mnist
    le_tests += [['mnist.c']]
    le_tests_link_with += mnist
endif

foreach filenames : le_tests
    filename = filenames[0]
    test_name = filename.split('.')[0].underscorify()
    exe = executable(test_name, filenames,
        include_directories: inc,
        link_with: le_tests_link_with,
        dependencies: glib,
        install: false
    )
    test(test_name, exe, timeout : 3 * 60)
endforeach

subdir('cpp')
subdir('python')
