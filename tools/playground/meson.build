# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

le_playground_sources = [
    'main.c',
    'pg-main-window.c',
    'pg-menus.c',
    'pg-generate-data.c',
    'pg-color.c'
]

gtk = dependency('gtk+-3.0',
    required: false
)

if gtk.found()
    executable('le-playground', le_playground_sources,
        dependencies: le_playground_deps,
        include_directories: incdir,
        link_with: le,
        install: true
    )
endif
