__device__ __forceinline__ float sigmoid(float a)
{
    return 1.0 / (1.0 + exp(-a));
}

__global__ void sigmoidKernel(float *a, int l)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < l; i += stride)
        a[i] = sigmoid(a[i]);
}

extern "C" void sigmoid_wrapper(float *a, int l)
{
    int blockSize = 256;
    int numBlocks = (l + blockSize - 1) / blockSize;
    sigmoidKernel<<<numBlocks, blockSize>>>(a, l);
}

__device__ __forceinline__ float sigmoid_prime(float a)
{
    float s = sigmoid(a);
    return s * (1.0f - s);
}

__global__ void sigmoidPrimeKernel(float *a, int l)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < l; i += stride)
        a[i] = sigmoid_prime(a[i]);
}

extern "C" void sigmoid_prime_wrapper(float *a, int l)
{
    int blockSize = 256;
    int numBlocks = (l + blockSize - 1) / blockSize;
    sigmoidPrimeKernel<<<numBlocks, blockSize>>>(a, l);
}