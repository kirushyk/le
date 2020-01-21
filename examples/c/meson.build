# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

executable('support-vector-machine', 'support-vector-machine.c',
    include_directories: incdir,
    link_with: le,
    install: false
)

executable('logistic-regression', 'logistic-regression.c',
    include_directories: incdir,
    link_with: le,
    install: false
)

executable('polynomia', 'polynomia.c',
    include_directories: incdir,
    link_with: le,
    install: false
)

executable('shallow-neural-network', 'shallow-neural-network.c',
    include_directories: incdir,
    link_with: le,
    install: false
)

subdir('mnist')
subdir('optimization')
