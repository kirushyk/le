#include <cstdint>
#include <le.hpp>

int main(int argc, char *argv[])
{
    std::uint32_t *sizes = new unsigned[4];
    for (int i = 0; i < 4; i++)
        sizes[i] = i + 1;
    le::Shape shape(4, sizes);
    return 0;
}
