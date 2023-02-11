cpp_tests = [
    'shape.cpp'
]

foreach filename : cpp_tests
    test_name = filename.split('.')[0].underscorify()
    exe = executable(test_name, filename,
        include_directories: [
            inc,
            inc_cpp
        ],
        link_with: [
            le_tests_deps,
            le_cpp
        ],
        install: false
    )
    test(test_name, exe, timeout : 3 * 60, suite: 'C++')
endforeach