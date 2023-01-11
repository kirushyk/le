kernel void hadamardProductKernel(device float *a[[buffer(0)]], const device float *b[[buffer(1)]], uint id[[thread_position_in_grid]])
{
    a[id] = a[id] * b[id];
}
