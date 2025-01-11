# Copyright (c) Kyrylo Polezhaiev
# Released under the MIT license. See LICENSE file in the project root for full license information.

tokenizer_sources = [
    'letokenizer.c'
]

tokenizer = static_library('le_tokenizer', tokenizer_sources,
    include_directories: inc,
    link_with: le,
    dependencies: [
        glib,
        dependency('json-glib-1.0')
    ]
)
