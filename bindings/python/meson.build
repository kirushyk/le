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

    py_platlib = py3.get_path('platlib')
    if not py_platlib.endswith('site-packages')
        error('Python3 platlib path seems invalid? ' + py_platlib)
    endif

    incdir = include_directories('../..', '../../bindings/cpp')

    pylib = py3.extension_module('le',
        'lemodule.cpp',
        link_with: [
            le,
            le_cpp
        ],
        dependencies: py_dep,
        include_directories: incdir,
        install: true
    )
endif

pypathdir = meson.current_build_dir()
