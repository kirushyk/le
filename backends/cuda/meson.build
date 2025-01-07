# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

le_cuda_sources = [
    'hadamard.cu',
    'sigmoid.cu',
    'lecuda.c'
]

have_cuda = add_languages(['cuda', 'cpp'], native: false, required: get_option('cuda')) 

cuda = dependency('cuda', version: '>=10', modules: ['cublas'], required: get_option('cuda'))
have_cuda = have_cuda and cuda.found()

le_cuda_deps = [
    glib,
    cuda
]

if have_cuda
    le_cuda = static_library('le_cuda', le_cuda_sources,
        include_directories: inc,
        dependencies: le_cuda_deps
    )
endif
