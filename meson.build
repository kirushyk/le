# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

project('le', 'c',
    version: '0.1.1',
    meson_version: '>= 1.0.0',
    default_options: [
        'c_std=c11',
        'cpp_std=c++14',
        'warning_level=1',
        'buildtype=debug'
    ]
)

cc = meson.get_compiler('c')
inc = include_directories('.')

subdir('platform')
subdir('le')
subdir('ext')
subdir('bindings')
subdir('tests')
subdir('benchmarks')

gtk = dependency('gtk4', required: false)

subdir('examples')
subdir('tools')
