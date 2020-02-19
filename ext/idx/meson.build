# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

idx_sources = [
    'leidx.c'
]

idx_deps = [
    cc.find_library('z')
]

idx = static_library('le_idx', idx_sources,
    dependencies: idx_deps,
    include_directories: incdir,
    link_with: le,
)
