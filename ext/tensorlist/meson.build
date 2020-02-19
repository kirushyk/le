# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

tensorlist_sources = [
    'letensorlist.c'
]

tensorlist = static_library('le_tensorlist', tensorlist_sources,
    include_directories: incdir,
    link_with: le,
)
