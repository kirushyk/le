le_metal_sources = [
    'lemetal.m'
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

have_metal = metal.found() and have_objc

if have_metal
    le_metal = static_library('le_metal', le_metal_sources,
        dependencies: [
            metal
        ],
        include_directories: include_directories('../../'),
    )
endif
