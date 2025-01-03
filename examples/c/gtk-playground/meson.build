# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

le_playground_sources = [
    'main.c',
    'pg-main-window.c',
    'pg-menus.c',
    'pg-generate-data.c',
    'pg-color.c'
]

if gtk.found()
    executable('le-playground', le_playground_sources,
        dependencies: gtk,
        include_directories: inc,
        link_with: [
            le,
            simple_dataset
        ],
        install: true
    )
endif
