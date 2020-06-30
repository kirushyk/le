#import "lemetal.h"
#import <le/le.h>
#import <le/tensors/letensor-imp.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

static id<MTLDevice> _Nonnull device;
static id<MTLCommandQueue> _Nonnull commandQueue;

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

    c->data = (void *)CFBridgingRetain([device newBufferWithLength:data_size options:0]);
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
//    cblas_sgemm(CblasRowMajor,
//                transpose_a ? CblasTrans : CblasNoTrans,
//                transpose_b ? CblasTrans : CblasNoTrans,
//                c_height, c_width, size_a,
//                1.0f,
//                a->data, a->shape->sizes[1],
//                b->data, b->shape->sizes[1],
//                0.0f,
//                c->data, c->shape->sizes[1]);
//
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

    tensor->data = (void *)CFBridgingRetain([device newBufferWithBytes:another->data length:data_size options:0]);
    
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
