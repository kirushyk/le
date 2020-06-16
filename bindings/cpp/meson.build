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

have_cpp = add_languages('cpp', required: false)
if have_cpp
    le_cpp = static_library('le_cpp',
        le_cpp_sources,
        include_directories: incdir,
        link_with: le
    )
endif
