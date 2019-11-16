# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

le_accelerate_sources = [
    'leaccelerate.c'
]

accelerate = dependency('appleframeworks', modules : 'accelerate')

le_accelerate_deps = [
    accelerate
]

if accelerate.found()
    le_accelerate = static_library('le_accelerate', le_accelerate_sources,
        dependencies: le_accelerate_deps,
        include_directories: incdir,
    )
endif
