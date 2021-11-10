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
    required: false
)

have_python = false
if py3.found()
    py_dep = py3.dependency(required: false)
    if have_cpp and py_dep.found()
        have_python = true
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

    pip3_pybind11_path = py3.get_install_dir() / 'pybind11' / 'include'
    if run_command('[', '-d', pip3_pybind11_path, ']').returncode() == 0
        pylib_inc += pip3_pybind11_path
    endif
    pip3_pybind11_path = py3.get_install_dir().replace('site-packages', 'dist-packages') / 'pybind11' / 'include'
    if run_command('[', '-d', pip3_pybind11_path, ']').returncode() == 0
        pylib_inc += pip3_pybind11_path
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
        dependencies: py_dep,
        include_directories: pylib_inc,
        install: true
    )
endif

pypathdir = meson.current_build_dir()
