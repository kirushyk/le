le_ext_tests = [
    'tokenizer'
]

foreach test_name : le_ext_tests
    subdir(test_name)
    test(test_name, ext_test,
        timeout : 3 * 60,
        suite: 'Ext',
    )
endforeach
