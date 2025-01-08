le_devices_libs = []

if have_cuda
    le_devices_libs += le_cuda
endif

if have_metal
    le_devices_libs += le_metal
endif

le_devices = static_library('ledevices', 'ledevices.c',
    dependencies: le_deps,
    include_directories: inc
)
