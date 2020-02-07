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
    'models/layers/lelayer.c',
    'models/layers/ledenselayer.c',
    'models/layers/leactivationlayer.c',
    'models/layers/leconv2d.c',
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

install_headers('lepolynomia.h', subdir : 'le')
install_headers('leloss.h', subdir : 'le')
install_headers('ledataset.h', subdir : 'le')
install_headers('letensor.h', subdir : 'le')
install_headers('optimization/leoptimizer.h', subdir : 'le/optimization')
install_headers('optimization/lebgd.h', subdir : 'le/optimization')
install_headers('lelist.h', subdir : 'le')
install_headers('leobject.h', subdir : 'le')
install_headers('models/lesequential.h', subdir : 'le/models')
install_headers('models/lesvm.h', subdir : 'le/models')
install_headers('models/layers/leconv2d.h', subdir : 'le/models/layers')
install_headers('models/layers/ledenselayer.h', subdir : 'le/models/layers')
install_headers('models/layers/leactivationlayer.h', subdir : 'le/models/layers')
install_headers('models/layers/lelayer.h', subdir : 'le/models/layers')
install_headers('models/lemodel.h', subdir : 'le/models')
install_headers('models/le1layernn.h', subdir : 'le/models')
install_headers('models/lelogistic.h', subdir : 'le/models')
install_headers('le.h', subdir : 'le')
install_headers('lemem.h', subdir : 'le')
install_headers('lematrix.h', subdir : 'le')
install_headers('letensor-imp.h', subdir : 'le')
install_headers('letype.h', subdir : 'le')
install_headers('lelog.h', subdir : 'le')
install_headers('leshape.h', subdir : 'le')

le = static_library('le', le_sources,
    include_directories: incdir,
    link_with: le_libs,
    dependencies: le_deps,
    install: true
)

pkg = import('pkgconfig')

pkg.generate(
    libraries: le,
    subdirs: [
        '.'
    ],
    version: '0.0.1',
    name: 'lible',
    filebase: 'le',
    description: 'Machine Learning library.',
    url: 'https://github.com/kirushyk/le'
)
