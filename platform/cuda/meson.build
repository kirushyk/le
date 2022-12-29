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

if have_cuda
    le_cuda = static_library('le_cuda', le_cuda_sources,
        dependencies: le_cuda_deps,
        include_directories: incdir,
    )
endif
