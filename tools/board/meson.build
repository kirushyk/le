# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

incdir = include_directories('../../')

le_board_sources = [
    'file.c',
    'main.c'
]

soup = dependency('libsoup-2.4',
    required: false
)

if soup.found()
    executable('le-board', le_board_sources,
        dependencies: soup,
        include_directories: incdir,
        link_with: le,
        install: true
    )
endif
