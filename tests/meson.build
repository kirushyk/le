le_tests = [
    'matrices.c'
]

foreach filename : le_tests
    test_name = filename.split('.')[0].underscorify()
    exe = executable(test_name, filename,
        include_directories: include_directories('..'),
        link_with: le,
        install: false
    )
    test(test_name, exe, timeout : 3 * 60)
endforeach
