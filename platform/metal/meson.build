le_metal_sources = [
    'lemetal.m'
]

le_metal_shaders = [
    'hadamard.metal'
]

have_objc = add_languages('objc', required: false)

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
    r = run_command(xcrun, '--sdk', 'macosx', '-f', 'metal')
    if r.returncode() == 0
        ## @todo: Prettify using dirs: argument
        metalc = find_program(r.stdout().strip(), required: false)
    else
        warning('unable to find metal utility')
    endif
    r = run_command(xcrun, '--sdk', 'macosx', '-f', 'metallib')
    if r.returncode() == 0
        metallib = find_program(r.stdout().strip(), required: false)
    else
        warning('unable to find metallib utility')
    endif
endif

have_metal = have_objc and metal.found()

if have_metal
    if metalc.found() and metallib.found()
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
        linked_shader = custom_target('le.metallib',
            output: 'le.metallib',
            input: compiled_shaders,
            command: [metallib, '@INPUT@', '-o', '@OUTPUT@']
        )
    else
        # Even if you have the command line tools installed, you still need Xcode.
        # You might have installed the command line tools before installing Xcode. For instance, you might have set up Homebrew first.
        # The command line tools are pointing to an incomplete SDK, rather than the one you installed with Xcode.
        # sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
    endif

    le_metal = static_library('le_metal', le_metal_sources + linked_shader,
        dependencies: [
            metal
        ],
        include_directories: include_directories('../../'),
    )
endif
