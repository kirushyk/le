__global__ void hadamardProductKernel(float *a, float *b, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        a[x * w + y] = a[x * w + y] * b[x * w + h];
    }
}

extern "C" void hadamard_wrapper(float *a, float *b, int w, int h) {
    hadamardProductKernel<<<1, 1>>>(a, b, w, h);
}

// kernel void hadamardProductKernel(device float *a[[buffer(0)]], const device float *b[[buffer(1)]], uint id[[thread_position_in_grid]])
// {
//     a[id] = a[id] * b[id];
// }
