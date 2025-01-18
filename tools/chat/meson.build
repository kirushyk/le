executable('le-chat', 'main.c',
    dependencies: le_deps + cc.find_library('readline'),
    link_with: [
        le,
        tokenizer,
    ],
    include_directories: inc
)
