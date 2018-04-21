# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

le_sources = [
    'letrainingdata.c',
    'lelogistic.c',
    'lematrix.c',
    'lesvm.c'
]

le_deps = [
    cc.find_library('m')
]

le = static_library('le', le_sources,
    dependencies: le_deps
)

