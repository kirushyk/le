#include <metal_math>

kernel void sigmoidKernel(device float *a[[buffer(0)]], uint id[[thread_position_in_grid]])
{
    a[id] = 1.0f / (1.0f + metal::exp(-a[id]));
}

kernel void sigmoidPrimeKernel(device float *a[[buffer(0)]], uint id[[thread_position_in_grid]])
{
    float s = 1.0f / (1.0f + metal::exp(-a[id]));
    a[id] = s * (1.0f - s);
}
