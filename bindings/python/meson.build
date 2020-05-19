py_mod = import('python')
py = py_mod.find_installation('python3')
py_dep = py.dependency()

pylib = py.extension_module('le',
    'lemodule.cpp',
    link_with: [
        le_cpp
    ],
    dependencies: [
        py_dep,
    ],
    install: true
)

pypathdir = meson.current_build_dir()
