# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

le_cuda_sources = [
    'lecuda.c',
    'hadamard.cu'
]

have_cuda = add_languages(['cuda', 'cpp'], required: get_option('cuda')) 

cuda = dependency('cuda', version: '>=10', modules: ['cublas'], required: get_option('cuda'))
have_cuda = have_cuda and cuda.found()

le_cuda_deps = [
    cuda
]

if have_cuda
    le_cuda = static_library('le_cuda', le_cuda_sources,
        dependencies: le_cuda_deps,
        include_directories: incdir
    )
endif
