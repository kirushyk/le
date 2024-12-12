# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

le_openblas_sources = [
    'leopenblas.c'
]

openblas = dependency('openblas', 
    modules: 'openblas',
    required: false
)

have_openblas = openblas.found()

if have_openblas
    le_openblas = static_library('le_openblas', le_openblas_sources,
        include_directories: inc,
        dependencies: [
            glib,
            openblas
        ]
    )
endif
