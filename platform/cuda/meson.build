# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

le_cuda_sources = [
    'lecuda.c'
]

cublas = dependency('cuda', version: '>=10', modules: ['cublas'], required: get_option('cuda'))

le_cuda_deps = [
    cublas
]

have_cuda = cublas.found()

le_cuda_dev_sources = [
    'hadamard.cu'
]

nvcc = find_program(['nvcc', '/usr/local/cuda/bin/nvcc'], required: false)
dev_objs = []
if nvcc.found()
    foreach dev_source: le_cuda_dev_sources
        obj_filename = dev_source.split('.')[0] + '.o'
        obj = custom_target(obj_filename,
            output: obj_filename,
            input: dev_source,
            command: [nvcc, '@INPUT@', '-c', '-o', '@OUTPUT@']
        )
        dev_objs += [obj]
    endforeach
endif

if have_cuda
    le_cuda = static_library('le_cuda', le_cuda_sources + dev_objs,
        dependencies: le_cuda_deps,
        include_directories: incdir,
    )
endif
