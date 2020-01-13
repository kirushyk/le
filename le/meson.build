# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../')

le_sources = [
    'lelist.c',
    'ledataset.c',
    'letype.c',
    'leshape.c',
    'letensor.c',
    'lematrix.c',
    'lepolynomia.c',
    'models/lelogistic.c',
    'models/le1layernn.c',
    'models/lemodel.c',
    'models/lesvm.c',
    'models/lelayer.c',
    'models/lesequential.c',
    'optimization/leoptimizer.c',
    'optimization/lebgd.c',
    'leloss.c',
    'lelog.c'
]

le_deps = [
    cc.find_library('m')
]

le_libs = []

if accelerate.found()
    le_libs += le_accelerate
endif

le = static_library('le', le_sources,
    include_directories: incdir,
    link_with: le_libs,
    dependencies: le_deps
)

