#import "lemetal.h"
#import <le/le.h>
#import <le/tensors/letensor-imp.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

static id<MTLDevice> device;
static id<MTLCommandQueue> commandQueue;

void
le_metal_init(void)
{
    device = MTLCreateSystemDefaultDevice();
    commandQueue = [device newCommandQueue];
}

LeTensor *
le_metal_matrix_new_product(const LeTensor *a, bool transpose_a, const LeTensor *b, bool transpose_b)
{
    assert(a->element_type == LE_TYPE_FLOAT32);
    assert(b->element_type == LE_TYPE_FLOAT32);
    assert(a->shape->num_dimensions == 2);
    assert(b->shape->num_dimensions == 2);
    assert(le_tensor_contiguous(a));
    assert(le_tensor_contiguous(b));
    
    unsigned size_a = transpose_a ? a->shape->sizes[0] : a->shape->sizes[1];
    unsigned size_b = transpose_b ? b->shape->sizes[1] : b->shape->sizes[0];
    assert(size_a == size_b);
    
    unsigned c_height = transpose_a ? a->shape->sizes[1] : a->shape->sizes[0];
    unsigned c_width = transpose_b ? b->shape->sizes[0] : b->shape->sizes[1];
        
    LeTensor *c = malloc(sizeof(struct LeTensor));
    c->device_type = LE_DEVICE_TYPE_METAL;
    c->element_type = LE_TYPE_FLOAT32;
    c->shape = le_shape_new(2, c_height, c_width);
    c->stride = le_shape_get_last_size(c->shape);
    c->owns_data = true;
    size_t data_size = le_shape_get_elements_count(c->shape) * le_type_size(c->element_type);
    
    id<MTLBuffer> buff_a = (__bridge id<MTLBuffer>)a->data;
    MPSMatrixDescriptor *desc_a =
        [MPSMatrixDescriptor matrixDescriptorWithRows: c_height
                                              columns: size_a
                                             rowBytes: a->stride * le_type_size(a->element_type)
                                             dataType: MPSDataTypeFloat32];
    MPSMatrix *mxa = [[MPSMatrix alloc] initWithBuffer: buff_a
                                            descriptor: desc_a];
    
    id<MTLBuffer> buff_b = (__bridge id<MTLBuffer>)b->data;
    MPSMatrixDescriptor *desc_b =
        [MPSMatrixDescriptor matrixDescriptorWithRows: size_b
                                              columns: c_width
                                             rowBytes: b->stride * le_type_size(b->element_type)
                                             dataType: MPSDataTypeFloat32];
    MPSMatrix *mxb = [[MPSMatrix alloc] initWithBuffer: buff_b
                                            descriptor: desc_b];
    
    id<MTLBuffer> buff_c = [device newBufferWithLength:data_size options:MTLResourceStorageModeShared];
    MPSMatrixDescriptor *desc_c =
        [MPSMatrixDescriptor matrixDescriptorWithRows: c_height
                                              columns: c_width
                                             rowBytes: c->stride * le_type_size(c->element_type)
                                             dataType: MPSDataTypeFloat32];
    MPSMatrix *mxc = [[MPSMatrix alloc] initWithBuffer: buff_c
                                            descriptor: desc_c];
    
    MPSMatrixMultiplication *kernel =
        [[MPSMatrixMultiplication alloc] initWithDevice: device
                                          transposeLeft: (BOOL)transpose_a
                                         transposeRight: (BOOL)transpose_b
                                             resultRows: (NSUInteger)c_height
                                          resultColumns: (NSUInteger)c_width
                                        interiorColumns: (NSUInteger)size_a
                                                  alpha: 1.0
                                                   beta: 0.0];
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    [kernel encodeToCommandBuffer: commandBuffer
                       leftMatrix: mxa
                      rightMatrix: mxb
                     resultMatrix: mxc];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    c->data = (void *)CFBridgingRetain(buff_c);
    
    return c;
}

LeTensor *
le_tensor_to_metal(const LeTensor *another)
{
    assert(another);
    assert(le_tensor_contiguous(another));
    assert(another->device_type == LE_DEVICE_TYPE_CPU);
    
    LeTensor *tensor = malloc(sizeof(struct LeTensor));
    tensor->device_type = LE_DEVICE_TYPE_METAL;
    tensor->element_type = another->element_type;
    tensor->shape = le_shape_copy(another->shape);
    tensor->stride = le_shape_get_last_size(tensor->shape);
    tensor->owns_data = true;
    size_t data_size = le_shape_get_elements_count(tensor->shape) * le_type_size(tensor->element_type);

    tensor->data = (void *)CFBridgingRetain([device newBufferWithBytes:another->data length:data_size options:MTLResourceStorageModeShared]);
    
    return tensor;
}

LeTensor *
le_tensor_to_cpu(const LeTensor *another)
{
    assert(another);
    assert(le_tensor_contiguous(another));
    assert(another->device_type == LE_DEVICE_TYPE_METAL);
    
    LeTensor *tensor = malloc(sizeof(struct LeTensor));
    tensor->device_type = LE_DEVICE_TYPE_CPU;
    tensor->element_type = another->element_type;
    tensor->shape = le_shape_copy(another->shape);
    tensor->stride = le_shape_get_last_size(tensor->shape);
    tensor->owns_data = true;
    size_t data_size = le_shape_get_elements_count(tensor->shape) * le_type_size(tensor->element_type);

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(another->data);
    tensor->data = malloc(data_size);
    memcpy(tensor->data, [buffer contents], data_size);
    
    return tensor;
}

void
le_metal_data_free(void *data)
{
    if (data)
    {
        CFBridgingRelease(data);
    }
}
