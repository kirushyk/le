pylib_sources = [
    'tensor.cpp',
    'svm.cpp',
    'lemodule.cpp'
]

py_mod = import('python')
py3 = py_mod.find_installation('python3',
    modules: [
        'pybind11'
    ],
    required: get_option('python')
)

py_deps = []

have_python = false
if py3.found()
    py_dep = py3.dependency(required: get_option('python'))
    if have_cpp and py_dep.found()
        have_python = true
        py_deps += py_dep
    endif
endif

if have_python
    py_purelib = py3.get_path('purelib')
    if not py_purelib.endswith('site-packages')
        error('Python3 purelib path seems invalid? ' + py_purelib)
    endif

    pylib_inc = [
        include_directories('../..', '../../bindings/cpp')
    ]
    pybind11_found = false
    pip3_pybind11_path = py3.get_install_dir() / 'pybind11' / 'include'
    if run_command('[', '-d', pip3_pybind11_path, ']').returncode() == 0
        pylib_inc += pip3_pybind11_path
        pybind11_found = true
    endif
    pip3_pybind11_path = py3.get_install_dir().replace('site-packages', 'dist-packages') / 'pybind11' / 'include'
    if not pybind11_found and run_command('[', '-d', pip3_pybind11_path, ']').returncode() == 0
        pylib_inc += pip3_pybind11_path
        pybind11_found = true
    endif
    if not pybind11_found
        py_deps += dependency('pybind11')
    endif

    py_platlib = py3.get_path('platlib')
    if not py_platlib.endswith('site-packages')
        error('Python3 platlib path seems invalid? ' + py_platlib)
    endif
    
    pylib = py3.extension_module('le',
        pylib_sources,
        link_with: [
            le,
            le_cpp
        ],
        dependencies: py_deps,
        include_directories: pylib_inc,
        install: true
    )
endif

pypathdir = meson.current_build_dir()
