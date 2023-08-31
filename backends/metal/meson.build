le_metal_sources = [
    'lemetal.m'
]

le_metal_shaders = [
    'sigmoid.metal',
    'hadamard.metal'
]

have_objc = add_languages('objc', native: false, required: false)

metal = dependency('appleframeworks',
    modules: [
        'foundation',
        'metal',
        'metalperformanceshaders'
    ],
    required: false
)

xcrun = find_program('xcrun', required: false)
metalc = disabler()
metallib = disabler()
if xcrun.found()
    r = run_command(xcrun, '--sdk', 'macosx', '-f', 'metal', check: false)
    if r.returncode() == 0
        ## @todo: Prettify using dirs: argument
        metalc = find_program(r.stdout().strip(), required: false)
    else
        warning('unable to find metal utility')
    endif
    r = run_command(xcrun, '--sdk', 'macosx', '-f', 'metallib', check: false)
    if r.returncode() == 0
        metallib = find_program(r.stdout().strip(), required: false)
    else
        warning('unable to find metallib utility')
    endif
endif

have_metal = have_objc and metal.found()

metallib_filename = 'le.metallib'

if have_metal
    if metalc.found() and metallib.found()
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
            dependencies: metal
        )
    else
        error('''Even if you have the command line tools installed, you still need Xcode.
You might have installed the command line tools before installing Xcode. For instance, you might have set up Homebrew first.
The command line tools are pointing to an incomplete SDK, rather than the one you installed with Xcode.
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer''')
    endif
endif
