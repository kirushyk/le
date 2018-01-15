incdir = include_directories('../../')

le_gtk_demo_sources = [
    'main.c',
    'main-window.c',
    'menus.c'
]

le_gtk_demo_deps = dependency('gtk+-3.0')

executable('le-gtk-demo', le_gtk_demo_sources,
    dependencies : le_gtk_demo_deps,
    include_directories: incdir,
    link_with: le,
    install: false
)

