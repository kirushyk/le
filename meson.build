# Copyright (c) 2017 Kyrylo V. Polezhaiev
# Released under the MIT license

project('le', 'c',
    version: '0.3.1',
    meson_version: '>= 1.0.0',
    default_options: [
        'c_std=c11',
        'cpp_std=c++14'
    ]
)

cc = meson.get_compiler('c')
inc = include_directories('.')
glib = dependency('glib-2.0')
gobject = dependency('gobject-2.0')

subdir('backends')
subdir('le')
subdir('ext')
subdir('bindings')
subdir('tests')
subdir('benchmarks')

subdir('examples')
subdir('tools')
