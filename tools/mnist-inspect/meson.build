# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

mnist_inspect_sources = [
    'main.c',
    'mi-main-window.c',
    'mi-menus.c'
]

mnist_inspect_deps = dependency('gtk+-3.0')

executable('mnist-inspect', mnist_inspect_sources,
    dependencies: mnist_inspect_deps,
    include_directories: incdir,
    link_with: [
        le,
        mnist
    ],
    install: true
)

