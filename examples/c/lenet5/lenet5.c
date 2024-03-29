#include <stdlib.h>
#include <le/le.h>

int main(int argc, char *argv[])
{
    LeSequential *nn = le_sequential_new();
    
    le_sequential_add(nn, LE_LAYER(le_conv2d_new ("L1", 5,
                          unsigned    num_channels,
                          unsigned    num_filters,
                          unsigned    padding,
                          unsigned    stride);("L1", 5, 3, 1)));
    le_sequential_add(nn, LE_LAYER(le_maxpool_layer_new("MP1", 2)));
    le_sequential_add(nn, LE_LAYER(le_conv2d_layer_new("L2", 5, 3, 1)));
    le_sequential_add(nn, LE_LAYER(le_maxpool_layer_new("MP2", 2)));

    return EXIT_SUCCESS;
}
