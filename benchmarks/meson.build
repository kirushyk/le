# Copyright (c) Kyrylo Polezhaiev and contributors. All rights reserved.
# Released under the MIT license. See LICENSE file in the project root for full license information.

le_benchmarks = [
    'matrices.c'
]

foreach filename : le_benchmarks
    test_name = filename.split('.')[0].underscorify() + '_benchmark'
        exe = executable(test_name, filename,
        include_directories: include_directories('..'),
        link_with: le,
        install: false
    )
    benchmark(test_name, exe, timeout : 3 * 60)
endforeach
