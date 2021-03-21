le_py_tests = [
    'svm.py'
]

if have_python
    foreach filename : le_py_tests
        test_name = filename.underscorify()
        test(test_name, py3, args: files(filename), timeout : 3 * 60, suite: 'Python')
    endforeach
endif
