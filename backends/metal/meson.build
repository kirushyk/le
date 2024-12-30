le_metal_sources = [
    'lemetal.m'
]

le_metal_shaders = [
    'sigmoid.metal',
    'hadamard.metal'
]

have_objc = add_languages('objc', native: false, required: get_option('metal'))

metal = dependency('appleframeworks',
    modules: [
        'foundation',
        'metal',
        'metalperformanceshaders'
    ],
    required: get_option('metal')
)

have_metal = false
metalc = disabler()
metallib = disabler()
if get_option('metal').disabled()
    if host_machine.system() == 'darwin'
        warning('Metal backend is explicitly disabled.')
    endif
    subdir_done()
endif

xcrun = find_program('xcrun', required: get_option('metal'))

if xcrun.found()
    r = run_command(xcrun, '--sdk', 'macosx', '-f', 'metal', check: false)
    if r.returncode() == 0
        ## @todo: Prettify using dirs: argument
        metalc = find_program(r.stdout().strip(), required: get_option('metal'))
    else
        warning('Unable to find `metal` command line tool.')
    endif
    r = run_command(xcrun, '--sdk', 'macosx', '-f', 'metallib', check: false)
    if r.returncode() == 0
        metallib = find_program(r.stdout().strip(), required: get_option('metal'))
    else
        warning('Unable to find `metallib` command line tool.')
    endif
else
    warning('Xcode Command Line Tools not found. Please install full Xcode application.')
endif

if have_objc and metal.found() and not (metalc.found() and metallib.found())
    warning('''Xcode Command Line Tools are pointing to an incomplete SDK instead of the one distributed with Xcode. This can happen due to one of the following reasons:
- Xcode is not installed: Even if you have the Xcode Command Line Tools installed, you still need the full Xcode application. Please install Xcode.
- Command Line Tools were installed before Xcode: For example, you may have set up Homebrew first. To fix this issue, run the following command:
    sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
To explicitly disable the Metal backend, configure the build folder with `-Dmetal=disabled`:
    meson configure -Dmetal=disabled build/''')
endif

have_metal = have_objc and metal.found() and metalc.found() and metallib.found()

metallib_filename = 'le.metallib'

if have_metal
    metal_config = configuration_data()
    compiled_shaders = []
    foreach shader: le_metal_shaders
        air_filename = shader.split('.')[0] + '.air'
        air = custom_target(air_filename,
            output: air_filename,
            input: shader,
            command: [metalc, '@INPUT@', '-c', '-o', '@OUTPUT@']
        )
        compiled_shaders += [air]
    endforeach
    metallib_path = get_option('datadir') / 'le' / 'metal'
    metal_config.set_quoted('METALLIB_FILENAME', metallib_filename)
    metal_config.set_quoted('METALLIB_BUILD_PATH', meson.current_build_dir())
    metal_config.set_quoted('METALLIB_INSTALL_PATH', get_option('prefix') / metallib_path)
    linked_shader = custom_target(metallib_filename,
        output: metallib_filename,
        input: compiled_shaders,
        command: [metallib, '@INPUT@', '-o', '@OUTPUT@'],
        install: true,
        install_dir: metallib_path
    )
    configure_file(output: 'metal-config.h', configuration: metal_config)

    # install_data(metallib_filename, install_dir: metallib_path)

    le_metal = static_library('le_metal', le_metal_sources,
        include_directories: inc,
        dependencies: [
            metal,
            glib
        ]
    )
elif get_option('metal').enabled()
    error('Metal backend is explicitly enabled but will not be built because Metal framework is not available on this system.')
elif host_machine.system() == 'darwin'
    warning('Metal backend will not be build, you may encounter suboptimal performance.')
endif
