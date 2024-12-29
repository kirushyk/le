#include <cuda_runtime.h>

__global__ void hadamardProductKernel(float *a, float *b, int l)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < l; i += stride)
        a[i] = a[i] * b[i];
}

extern "C" void hadamard_wrapper(float *a, float *b, int l)
{
    int blockSize = 256;
    int numBlocks = (l + blockSize - 1) / blockSize;
    hadamardProductKernel<<<numBlocks, blockSize>>>(a, b, l);
}
