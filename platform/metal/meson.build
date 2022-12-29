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

have_metal = have_objc and metal.found() and metalc.found() and metallib.found()

if have_metal
    compiled_shaders = []
    foreach shader : le_metal_shaders
        air_filename = shader.split('.')[0] + '.air'
        air = custom_target(air_filename,
            output : air_filename,
            input : shader,
            command : [metalc, '@INPUT@', '-o', '@OUTPUT@']
        )
        message(air_filename)
        linked_shader_filename = shader.split('.')[0] + '.metallib'
        linked_shader = custom_target(linked_shader_filename,
            output : linked_shader_filename,
            input : air,
            command : [metallib, '@INPUT@', '-o', '@OUTPUT@']
        )
        compiled_shaders += [linked_shader]
    endforeach

    le_metal = static_library('le_metal', le_metal_sources,
        dependencies: [
            metal
        ],
        include_directories: include_directories('../../'),
    )
endif
