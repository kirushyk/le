py_mod = import('python')
py3 = py_mod.find_installation('python3',
    modules: [
        'pybind11'
    ]
)
py_purelib = py3.get_path('purelib')
if not py_purelib.endswith('site-packages')
  error('Python3 purelib path seems invalid? ' + py_purelib)
endif

py_platlib = py3.get_path('platlib')
if not py_platlib.endswith('site-packages')
  error('Python3 platlib path seems invalid? ' + py_platlib)
endif
py_dep = py3.dependency()

pylib = py3.extension_module('le',
    'lemodule.cpp',
    link_with: le_cpp,
    dependencies: py_dep,
    install: true
)

pypathdir = meson.current_build_dir()
