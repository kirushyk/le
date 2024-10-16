le_cpp_sources = [
    'tensor.cpp',
    'model.cpp',
    'logistic.cpp',
    'svm.cpp',
    'shape.cpp',
    'sequential.cpp',
    'layer.cpp',
    'activation-layer.cpp',
    'dense-layer.cpp',
    'optimizer.cpp',
    'bgd.cpp'
]

inc_cpp = include_directories('.')

have_cpp = add_languages('cpp', native: false, required: false)
if have_cpp
    le_cpp = static_library('le_cpp',
        le_cpp_sources,
        include_directories: [
            inc,
            inc_cpp
        ],
        link_with: le,
        dependencies: [
            glib,
            gobject
        ]
    )
endif
