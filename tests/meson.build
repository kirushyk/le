# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

config = configuration_data()
config.set_quoted('TEST_DIR', meson.current_build_dir())
configure_file(output: 'test-config.h', configuration: config)

le_tests = [
    ['objects/objects.c', 'objects/base.c', 'objects/a.c', 'objects/b.c'],
    ['list.c'],
    ['matrices.c'],
    ['type-generic.c'],
    ['sobel.c'],
    ['relu.c'],
    ['tensorlist.c'],
    ['subtensor.c'],
    ['input_normalization.c'],
    ['cnn-inf.c']
    # ['gradcheck.c']
]

le_tests_deps = [
    le,
    tensorlist
]

if have_metal
    le_tests += [
        #['metal/hadamard.c'],
        ['metal/matmul.c']
    ]
    le_tests_deps += le_metal
endif

if have_mnist
    le_tests += [['mnist.c']]
    le_tests_deps += mnist
endif

foreach filenames : le_tests
    filename = filenames[0]
    test_name = filename.split('.')[0].underscorify()
    exe = executable(test_name, filenames,
        include_directories: include_directories('..'),
        link_with: le_tests_deps,
        install: false
    )
    test(test_name, exe, timeout : 3 * 60)
endforeach

subdir('cpp')
subdir('python')
