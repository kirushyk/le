# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

idx_sources = [
    'leidx.c'
]

zlib = cc.find_library('z', required: false)

have_idx = zlib.found()

if zlib.found()
    idx_deps = [
        zlib
    ]

    idx = static_library('le_idx', idx_sources,
        include_directories: incdir,
        dependencies: idx_deps,
        link_with: le,
    )
endif
