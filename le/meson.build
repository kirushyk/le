# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

le_sources = [
    'lelist.c',
    'ledataset.c',
    'letype.c',
    'leshape.c',
    'letensor.c',
    'lematrix.c',
    'lepolynomia.c',
    'lelogistic.c',
    'lemodel.c',
    'lesvm.c',
    'lesequential.c'
]

le_deps = [
    cc.find_library('m')
]

le = static_library('le', le_sources,
    dependencies: le_deps
)

