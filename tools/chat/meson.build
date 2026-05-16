readline = cc.find_library('readline', required: false)

if readline.found()
    executable('le-chat', 'main.c',
        dependencies: le_deps + readline,
        link_with: [
            le,
            tokenizer,
        ],
        include_directories: inc
    )
endif
