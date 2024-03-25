# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

le_sources = [
    'ledataset.c',
    'math/lerand.c',
    'math/leclamp.c',
    'math/lepolynomia.c',
    'tensors/letype.c',
    'tensors/leshape.c',
    'tensors/letensor.c',
    'tensors/letensor-cast.c',
    'tensors/lescalar.c',
    'tensors/lematrix.c',
    # 'models/leknn.c',
    'models/lelogistic.c',
    # 'models/le1layernn.c',
    'models/lemodel.c',
    'models/lesvm.c',
    'models/layers/lelayer.c',
    'models/layers/ledenselayer.c',
    'models/layers/leactivationlayer.c',
    # 'models/layers/leconv2d.c',
    'models/lesequential.c',
    'optimization/leoptimizer.c',
    'optimization/lebgd.c',
    # 'optimization/lesgd.c',
    # 'optimization/adam.c',
    'leloss.c',
    'lelog.c'
]

le_deps = [
    cc.find_library('m'),
    glib,
    gobject
]

le_libs = []

if accelerate.found()
    le_libs += le_accelerate
endif

if have_openblas
    le_libs += le_openblas
endif

if have_cuda
    le_libs += le_cuda
endif

if have_metal
    le_libs += le_metal
endif

config = configuration_data()
config.set('HAVE_OPENBLAS', have_openblas)
config.set('HAVE_CUDA', have_cuda)
config.set('HAVE_METAL', have_metal)
configure_file(output: 'config.h', configuration: config)

install_headers('leloss.h', subdir : 'le')
install_headers('ledataset.h', subdir : 'le')
install_headers('math/lepolynomia.h', subdir : 'le')
install_headers('math/lerand.h', subdir : 'le/math')
install_headers('math/leclamp.h', subdir : 'le/math')
install_headers('tensors/letensor.h', subdir : 'le/tensors')
install_headers('optimization/leoptimizer.h', subdir : 'le/optimization')
install_headers('optimization/lebgd.h', subdir : 'le/optimization')
install_headers('optimization/lesgd.h', subdir : 'le/optimization')
install_headers('optimization/adam.h', subdir : 'le/optimization')
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
install_headers('tensors/letype.h', subdir : 'le')
install_headers('tensors/leshape.h', subdir : 'le')
install_headers('tensors/lematrix.h', subdir : 'le/tensors')
install_headers('tensors/letensor-imp.h', subdir : 'le/tensors')
install_headers('tensors/letensor-cast.h', subdir : 'le/tensors')
install_headers('tensors/lescalar.h', subdir : 'le/tensors')
install_headers('lelog.h', subdir : 'le')

le = library('le', le_sources,
    include_directories: inc,
    link_with: le_libs,
    dependencies: le_deps,
    install: true
)

pkg = import('pkgconfig')

le_pkg_libs = [
    le
]

if accelerate.found()
    le_pkg_libs += accelerate
endif

if openblas.found()
    le_pkg_libs += openblas
endif

pkg.generate(
    libraries: le_pkg_libs,
    subdirs: [
        '.'
    ],
    version: '0.0.1',
    name: 'lible',
    filebase: 'le',
    description: 'Machine Learning library.',
    url: 'https://github.com/kirushyk/le'
)
