#include <metal_stdlib>
using namespace metal;

kernel void hadamardProductKernel(
    texture_buffer<float, access::read_write> A [[texture(0)]],
    texture_buffer<float, access::read> B [[texture(1)]],
    uint gid [[thread_position_in_grid]]
)
{
    A.write(A.read(gid) * B.read(gid), gid);
}
