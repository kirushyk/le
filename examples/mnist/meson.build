mnist_example_deps = [
    le,
    mnist
]

executable('mnist-logistic-regression', 'mnist-logistic-regression.c',
    include_directories: incdir,
    link_with: mnist_example_deps,
    install: false
)

executable('mnist-1-layer-nn', 'mnist-1-layer-nn.c',
    include_directories: incdir,
    link_with: mnist_example_deps,
    install: false
)
