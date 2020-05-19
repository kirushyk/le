#include <le.hpp>

int main(int argc, char *argv[])
{
    le::Tensor x(le::Type::FLOAT32, 2, 2, 4,
        1.0, 2.0, 1.0, 2.0,
        2.0, 2.0, 1.0, 1.0
    );

    le::Tensor y(le::Type::FLOAT32, 2, 1, 4,
        0.0, 1.0, 1.0, 0.0
    );

    return 0;
}
