# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

le_cuda_sources = [
    'lecuda.cpp'
]

cublas = dependency('cuda', version: '>=10', modules: ['cublas'], required: false)

le_cuda_deps = [
    cublas
]

if cublas.found()
    le_accelerate = static_library('le_accelerate', le_cuda_sources,
        dependencies: le_cuda_deps,
        include_directories: incdir,
    )
endif
