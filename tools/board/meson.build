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
    board_ui_path = get_option('datadir') / 'le' / 'tools' / 'board'

    config = configuration_data()
    config.set_quoted('BOARD_UI_SOURCE_PATH', meson.current_source_dir() / 'ui')
    config.set_quoted('BOARD_UI_INSTALL_PATH', get_option('prefix') / board_ui_path)
    configure_file(output: 'board-config.h', configuration: config)

    install_data(
        [
            'ui/bootstrap.min.css',
            'ui/bootstrap.min.js',
            'ui/index.html',
            'ui/moment.min.js',
            'ui/ui.js'
        ],
        install_dir : board_ui_path / 'ui'
    )

    executable('le-board', le_board_sources,
        dependencies: soup,
        include_directories: incdir,
        link_with: le,
        install: true
    )
endif
