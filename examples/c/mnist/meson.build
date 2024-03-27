if have_mnist
    mnist_example_deps = [
        le,
        mnist,
        tensorlist
    ]

    executable('mnist-logistic-regression', 'mnist-logistic-regression.c',
        include_directories: inc,
        link_with: mnist_example_deps,
        install: false,
        dependencies: [glib, gobject]
    )

    executable('mnist-1-layer-nn', 'mnist-1-layer-nn.c',
        include_directories: inc,
        link_with: mnist_example_deps,
        install: false,
        dependencies: [glib, gobject]
    )

    executable('mnist-snn', 'mnist-snn.c',
        include_directories: inc,
        link_with: mnist_example_deps,
        install: false,
        dependencies: [glib, gobject]
    )

    executable('mnist-2nn', 'mnist-2nn.c',
        include_directories: inc,
        link_with: mnist_example_deps,
        install: false,
        dependencies: [glib, gobject]
    )
endif
