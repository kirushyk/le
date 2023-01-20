# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

le_cuda_sources = [
    'lecuda.c'
]

stdcpp = meson.get_compiler('cpp').find_library('stdc++', required: get_option('cuda'))
cuda = dependency('cuda', version: '>=10', modules: ['cublas'], required: get_option('cuda'))
have_cuda = cuda.found() and stdcpp.found()

le_cuda_deps = [
    stdcpp,
    cuda
]

le_cuda_dev_sources = [
    'hadamard.cu'
]

if have_cuda
    nvcc = find_program(['nvcc', '/usr/local/cuda/bin/nvcc'], required: false)
    dev_objs = []
    if nvcc.found()
        foreach dev_source: le_cuda_dev_sources
            obj_filename = dev_source.split('.')[0] + '.o'
            obj = custom_target(obj_filename,
                output: obj_filename,
                input: dev_source,
                command: [
                    nvcc, '@INPUT@',
                    '-I../', '-Xcompiler', '\'-fPIC\'',
                    '-c', '-o', '@OUTPUT@'
                ]
            )
            dev_objs += [obj]
        endforeach
    endif

    le_cuda = static_library('le_cuda', le_cuda_sources + dev_objs,
        dependencies: le_cuda_deps,
        include_directories: incdir
    )
endif
