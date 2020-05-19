py_mod = import('python')
py = py_mod.find_installation()
py_dep = py.dependency()

pylib = py.extension_module('le',
    'lemodule.cpp',
    dependencies: [
        le_cpp,
        py_dep,
    ]
)

pypathdir = meson.current_build_dir()
