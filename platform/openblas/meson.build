# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

le_openblas_sources = [
    'leopenblas.c'
]

openblas = dependency('openblas', 
    modules: 'openblas',
    required: false
)

if openblas.found()
    le_openblas = static_library('le_openblas', le_openblas_sources,
        dependencies: [
            openblas
        ],
        include_directories: incdir,
    )
endif
