#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

void le_metal_init(void)
{
    id<MTLDevice> _Nonnull device;
    id<MTLCommandQueue> _Nonnull commandQueue;
    device = MTLCreateSystemDefaultDevice();
    commandQueue = [device newCommandQueue];
}
