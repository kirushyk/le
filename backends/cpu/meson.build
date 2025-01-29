le_cpu_sources = [
    'lecpu.c'
]

le_cpu_backend_deps = [
    glib
]

le_cpu_backend_c_args = []

accelerate_framework = dependency('appleframeworks', 
    modules: 'accelerate',
    required: false
)

if accelerate_framework.found()
    le_cpu_backend_deps += accelerate_framework
    le_cpu_backend_c_args += [
        '-DACCELERATE_NEW_LAPACK',
        '-DACCELERATE_LAPACK_ILP64'
    ]
endif

openblas = dependency('openblas', 
    modules: 'openblas',
    required: false
)

if openblas.found()
    le_cpu_backend_deps += openblas
endif

config = configuration_data()
config.set('HAVE_OPENBLAS', openblas.found())
config.set('HAVE_ACCELERATE_FRAMEWORK', accelerate_framework.found())
configure_file(output: 'cpu-backend-config.h', configuration: config)

le_cpu_backend = static_library('le_cpu', le_cpu_sources,
    include_directories: inc,
    c_args: le_cpu_backend_c_args,
    dependencies: le_cpu_backend_deps,
)
