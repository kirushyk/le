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
    'le.cpp'
]

if add_languages('cpp', native: false, required: false)
    le_cpp = static_library('le_cpp',
        le_cpp_sources,
        include_directories: incdir,
        link_with: le
    )
endif
