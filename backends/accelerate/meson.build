# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

le_accelerate_sources = [
    'leaccelerate.c'
]

accelerate = dependency('appleframeworks', 
    modules: 'accelerate',
    required: false
)

if accelerate.found()
    le_accelerate = static_library('le_accelerate', le_accelerate_sources,
        include_directories: inc,
        c_args: [
            '-DACCELERATE_NEW_LAPACK',
            '-DACCELERATE_LAPACK_ILP64',
        ],
        dependencies: [
            accelerate,
            glib
        ]
    )
endif
