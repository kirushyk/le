# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

mnist_inspect_sources = [
    'main.c',
    'mi-main-window.c',
    'mi-menus.c'
]

if gtk.found()
    executable('mnist-inspect', mnist_inspect_sources,
        dependencies: gtk,
        include_directories: inc,
        link_with: [
            le,
            mnist
        ],
        install: true
    )
endif
