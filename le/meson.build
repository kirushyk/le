le_sources = [
    'letrainingdata.c',
    'lelogistic.c',
    'lematrix.c',
    'lesvm.c'
]

le_deps = [
    cc.find_library('m')
]

le = static_library('le', le_sources,
    dependencies: le_deps
)

