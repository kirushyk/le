# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

project('le', 'c',
    version: '0.0.1',
    meson_version: '>= 0.40.1',
    default_options: [
        'c_std=c99',
        'warning_level=1',
        'buildtype=debug'
    ]
)

cc = meson.get_compiler('c')

subdir('le')
subdir('ext')
subdir('tests')
subdir('benchmarks')
subdir('examples')
subdir('tools')
