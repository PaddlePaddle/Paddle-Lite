// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/backends/metal/metal_context_imp.h"
#include "lite/kernels/metal/image_op/fetch_image_compute.h"
#include "lite/utils/log/cp_logging.h"

// debug macro
//#define METAL_DEBUG_GPU_CAPTURE
//#define METAL_DEBUG_ONE_COMMANDBUFFER

extern NSString* cString2NSString(std::string cStr) {
    return [NSString stringWithCString:cStr.c_str() encoding:[NSString defaultCStringEncoding]];
}

@interface MetalContextImp () {
    bool use_one_cmdbuf_;
    bool use_memory_reuse_;
    std::vector<paddle::lite::kernels::metal::FetchImageCompute*> _fetch_vector;
}
@property (strong, nonatomic) NSString* libPath;
@property (strong, nonatomic) id<MTLDevice> device;
@property (strong, nonatomic) id<MTLLibrary> library;
@property (strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property (strong, nonatomic) id<MTLCommandBuffer> commandBuffer;
@property (strong, nonatomic) NSMutableArray<id<MTLCommandBuffer>>* waitings;
@property (strong, nonatomic) NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* caches;
@property (strong, nonatomic) NSMutableDictionary* memoryReuseHeaps API_AVAILABLE(ios(10.0));
@end

@implementation MetalContextImp

#pragma mark - external

- (instancetype)init {
    self = [super init];
    if (self) {
        _fetch_vector = {};
        _waitings = [NSMutableArray array];
        _caches = [NSMutableDictionary dictionary];
        _memoryReuseHeaps = [NSMutableDictionary dictionaryWithCapacity:3];
        _device = MTLCreateSystemDefaultDevice();
        _commandQueue = [_device newCommandQueue];
#if defined(METAL_DEBUG_GPU_CAPTURE)
        // At least one command buffer must be created and
        // committed within the boundaries of a GPU Capture.
        [self startCapture];
#endif
#if defined(METAL_DEBUG_ONE_COMMANDBUFFER)
        use_one_cmdbuf_ = true;
#endif
        _commandBuffer = [_commandQueue commandBuffer];
    }
    return self;
}

- (void)dealloc {
    _fetch_vector.clear();
}

- (void)setMetalPath:(std::string)path {
    NSString* pathStr = cString2NSString(path);
    NSError* libraryErr = nil;
    if (pathStr) {
        self.libPath = pathStr;
        self.library = [self.device newLibraryWithFile:pathStr error:&libraryErr];
    }
    if (nil == _library) {
        LOG(ERROR) << "Can't load metallib: "
                   << [pathStr cStringUsingEncoding:NSUTF8StringEncoding];
    }
}

- (void)setMetalDevice:(void*)device {
    if (!device) {
        return;
    }
    id<MTLDevice> mtl = (__bridge id<MTLDevice>)device;
    if (mtl) {
        self.device = mtl;
        self.commandQueue = [_device newCommandQueue];
#if defined(METAL_DEBUG_GPU_CAPTURE)
        [self startCapture];
#endif
        self.commandBuffer = [self.commandQueue commandBuffer];
    }
    if (self.libPath) {
        [self setMetalPath:self.libPath.UTF8String];
    }
}

#pragma mark data

- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size access:(paddle::lite::METAL_ACCESS_FLAG)access {
    return [_device newBufferWithLength:size options:[self optionForAccess:access]];
}

- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size
                           bytes:(void*)bytes
                          access:(paddle::lite::METAL_ACCESS_FLAG)access {
    return [_device newBufferWithBytes:bytes length:size options:[self optionForAccess:access]];
}

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor*)desc {
    id<MTLTexture> image = [_device newTextureWithDescriptor:desc];
    assert(nil != image);
    return image;
}

#pragma mark enqueue

- (id<MTLComputePipelineState>)pipline:(std::string)name {
    NSString* nameStr = cString2NSString(name);
    if (!nameStr) {
        return nil;
    }

    id<MTLFunction> function = [_library newFunctionWithName:nameStr];
    if (!function) {
        return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipline =
        [_device newComputePipelineStateWithFunction:function error:&error];
    assert(nil != pipline);
    return pipline;
}

- (id<MTLComputeCommandEncoder>)commandEncoder {
    id<MTLComputeCommandEncoder> result = [_commandBuffer computeCommandEncoder];
    assert(nil != result);
    return result;
}

- (void)commit {
    if (use_one_cmdbuf_) {
    } else {
        [_commandBuffer commit];
        [_waitings addObject:_commandBuffer];
        _commandBuffer = [_commandQueue commandBuffer];
    }
}

// mps
- (id<MTLCommandBuffer>)commandBuffer {
    if (use_one_cmdbuf_) {
        return _commandBuffer;
    } else {
        id<MTLCommandBuffer> result = [_commandQueue commandBuffer];
        assert(nil != result);
        return result;
    }
}

// mps
- (void)commit:(id<MTLCommandBuffer>)cmdBuf {
    assert(nil != cmdBuf);
    if (use_one_cmdbuf_) {
    } else {
        [cmdBuf commit];
        [_waitings addObject:cmdBuf];
    }
}

- (void)waitAllCompleted {
    if (use_one_cmdbuf_) {
        [_commandBuffer commit];
#if defined(METAL_DEBUG_GPU_CAPTURE)
        [self stopCapture];
#endif
        [_commandBuffer waitUntilCompleted];
        _commandBuffer = [_commandQueue commandBuffer];
    } else {
#if defined(METAL_DEBUG_GPU_CAPTURE)
        [self stopCapture];
#endif
        for (id<MTLCommandBuffer> buffer in _waitings) {
            if (buffer.status >= MTLCommandBufferStatusCompleted) {
                continue;
            }
            [buffer waitUntilCompleted];
            if (buffer.error) {
                VLOG(4) << "[METAL]: " << buffer.error.localizedDescription.UTF8String;
            }
        }
        [_waitings removeAllObjects];
    }
}

#pragma mark c++ external

- (void)add_fetch_kernel_ptr:(void*)ptr {
    paddle::lite::kernels::metal::FetchImageCompute* fetch =
        static_cast<paddle::lite::kernels::metal::FetchImageCompute*>(ptr);
    _fetch_vector.push_back(fetch);
}

- (void)fetch_data_from_gpu {
    for (auto item : _fetch_vector) {
        item->fetch_data_from_gpu();
    }
}

#pragma mark dispatch

- (void)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                pipline:(id<MTLComputePipelineState>)pipline
             outTexture:(id<MTLTexture>)outTexture {
    [self dispatchEncoder:encoder pipline:pipline outTexture:outTexture quadruple:false];
}

- (void)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                pipline:(id<MTLComputePipelineState>)pipline
             outTexture:(id<MTLTexture>)outTexture
              quadruple:(bool)quadruple {
    if (!pipline) {
        LOG(ERROR) << "[metal] invalid param, MTL Compute Pipeline is nil!";
        return;
    }
    NSUInteger slices = (outTexture.arrayLength * 4 + 3) / 4;
    NSUInteger width = 0, height = 0, groupWidth = 0, groupHeight = 0;
    if (quadruple) {
        width = pipline.threadExecutionWidth / 4;
        width = MIN(width, outTexture.width);
        height = pipline.threadExecutionWidth / width;
        height = MIN(height, outTexture.height);
        groupWidth = (outTexture.width / 2 + width - 1) / width;
        groupHeight = (outTexture.height / 2 + height - 1) / height;
    } else {
        width = pipline.threadExecutionWidth;
        width = MIN(width, outTexture.width);
        height = pipline.maxTotalThreadsPerThreadgroup / width;
        height = MIN(height, outTexture.height);
        groupWidth = (outTexture.width + width - 1) / width;
        groupHeight = (outTexture.height + height - 1) / height;
    }
    MTLSize threadsPerGroup = MTLSize{.width = width, .height = height, .depth = 1};
    MTLSize groups = MTLSize{.width = groupWidth, .height = groupHeight, .depth = slices};
    if (groups.width <= 0 || groups.height <= 0 || groups.depth <= 0) {
        LOG(ERROR) << "[METAL]: "
                   << "dispatch thread groups 1.{" << groups.width << "," << groups.height << ","
                   << groups.depth << "}";
        return;
    }
    [encoder setComputePipelineState:pipline];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
}

// threadsShape: [n, h, w]
- (void)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                pipline:(id<MTLComputePipelineState>)pipline
           threadsShape:(NSArray<NSNumber*>*)threadsShape {
    if ([threadsShape count] != 3) {
        LOG(ERROR) << "[metal] invalid param, MTL Compute Pipeline is nil!";
        return;
    }

    NSUInteger tZ = threadsShape[0].integerValue;
    NSUInteger tH = threadsShape[1].integerValue;
    NSUInteger tW = threadsShape[2].integerValue;

    NSUInteger slices = (tZ + 3) / 4;
    NSUInteger width = 0, height = 0;
    width = MIN(pipline.threadExecutionWidth, tW);
    height = MIN(pipline.maxTotalThreadsPerThreadgroup / width, tH);
    MTLSize threadsPerGroup = MTLSize{.width = width, .height = height, .depth = 1};

    NSUInteger groupWidth = 0, groupHeight = 0;
    groupWidth = (tW + width - 1) / width;
    groupHeight = (tH + height - 1) / height;
    MTLSize groups = MTLSize{.width = groupWidth, .height = groupHeight, .depth = slices};

    [self dispatchEncoder:encoder pipline:pipline threadsPerGroup:threadsPerGroup groups:groups];
}

- (void)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                pipline:(id<MTLComputePipelineState>)pipline
        threadsPerGroup:(MTLSize)threadsPerGroup
                 groups:(MTLSize)groups {
    if (!pipline) {
        LOG(ERROR) << "[metal] invalid param, MTL Compute Pipeline is nil!";
        return;
    }
    if (groups.width <= 0 || groups.height <= 0 || groups.depth <= 0) {
        LOG(ERROR) << "[METAL]: "
                   << "dispatch thread groups 2.{" << groups.width << "," << groups.height << ","
                   << groups.depth << "}";
        return;
    }
    [encoder setComputePipelineState:pipline];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
}

#pragma mark pre-process

- (MPSImageLanczosScale*)lanczosScalePtrCreate {
    MPSImageLanczosScale* lanczos = [[MPSImageLanczosScale alloc] initWithDevice:self.device];
    return lanczos;
}

- (id<MTLTexture>)lanczosTextureCreate:(NSArray*)dims {
    MTLTextureDescriptor* textureDesc = [[MTLTextureDescriptor alloc] init];
    textureDesc.textureType = MTLTextureType2D;
    textureDesc.width = [dims[3] intValue];
    textureDesc.height = [dims[2] intValue];
    textureDesc.depth = ([dims[1] intValue] + 3) / 4;
    textureDesc.pixelFormat = MTLPixelFormatRGBA16Float;
    textureDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    ;
    textureDesc.storageMode = MTLStorageModeShared;

    id<MTLTexture> texture = [self.device newTextureWithDescriptor:textureDesc];
    return texture;
}

#pragma mark memory reuse

- (void)set_use_memory_reuse:(bool)flag {
    use_memory_reuse_ = flag;
    if (flag) {
        use_one_cmdbuf_ = true;
    }
}

- (void)setHeap:(id<MTLHeap>)heap key:(std::string)ptr API_AVAILABLE(ios(10.0)) {
    NSString* ptrStr = cString2NSString(ptr);
    if (!ptrStr) {
        LOG(INFO) << "heap key is nil";
        return;
    }
    [self.memoryReuseHeaps setObject:heap forKey:ptrStr];
}

- (id<MTLHeap>)getHeap:(std::string)ptr API_AVAILABLE(ios(10.0)) {
    NSString* ptrStr = cString2NSString(ptr);
    if (!ptrStr) {
        return nil;
    }
    if ([self.memoryReuseHeaps objectForKey:ptrStr]) {
        return [self.memoryReuseHeaps objectForKey:ptrStr];
    }
    return nil;
}

- (id<MTLHeap>)newHeapWithDescriptor:(MTLTextureDescriptor*)desc API_AVAILABLE(ios(10.0)) {
    if (@available(iOS 10.0, *)) {
        NSInteger size = [_device heapTextureSizeAndAlignWithDescriptor:desc].size;
        MTLHeapDescriptor* heapDesc = [[MTLHeapDescriptor alloc] init];
        heapDesc.size = size;
        heapDesc.storageMode = MTLStorageModeShared;
        return [_device newHeapWithDescriptor:heapDesc];
    }
    return nil;
}

- (bool)isNewHeapWithDescriptor:(MTLTextureDescriptor*)desc
                           heap:(id<MTLHeap>)heap API_AVAILABLE(ios(10.0)) {
    if (@available(iOS 10.0, *)) {
        NSInteger hSize = heap.size;
        NSInteger tSize = [_device heapTextureSizeAndAlignWithDescriptor:desc].size;
        if (hSize >= tSize) {
            return false;
        } else {
            return true;
        }
    }
    return false;
}

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor*)desc
                                      heap:(id<MTLHeap>)heap API_AVAILABLE(ios(10.0)) {
    if (@available(iOS 10.0, *)) {
        id<MTLTexture> image = [heap newTextureWithDescriptor:desc];
        [image makeAliasable];
        assert(nil != image);
        return image;
    }
    return nil;
}

#pragma mark - internal

- (MTLResourceOptions)optionForAccess:(paddle::lite::METAL_ACCESS_FLAG)access {
    if (access == paddle::lite::METAL_ACCESS_FLAG::CPUWriteOnly) {
        return MTLResourceOptionCPUCacheModeWriteCombined;
    } else if (access == paddle::lite::METAL_ACCESS_FLAG::CPUTransparent) {
        if (@available(iOS 9.0, *)) {
            return MTLResourceStorageModePrivate;
        } else {
            return MTLResourceOptionCPUCacheModeDefault;
        }
    } else if (access == paddle::lite::METAL_ACCESS_FLAG::CPUShared) {
        return MTLStorageModeShared;
    } else {  // access == paddle::lite::METAL_ACCESS_FLAG::CPUReadWrite
        return MTLResourceOptionCPUCacheModeDefault;
    }
}

- (id<MTLFunction>)functionWithName:(NSString*)name {
    if (!name) {
        return nil;
    }
    id<MTLFunction> result = [_library newFunctionWithName:name];
    return result;
}

- (id<MTLComputePipelineState>)pipelineWithName:(NSString*)name {
    id<MTLComputePipelineState> result = _caches[name];
    if (result) {
        return result;
    }

    id<MTLFunction> function = [self functionWithName:name];
    if (!function) {
        return nil;
    }

    NSError* error = nil;
    result = [_device newComputePipelineStateWithFunction:function error:&error];
    if (result) {
        _caches[name] = result;
    }
    return result;
}

#pragma mark - capture

- (void)startCapture {
    if (@available(iOS 13.0, *)) {
        MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
        MTLCaptureDescriptor* captureDescriptor = [[MTLCaptureDescriptor alloc] init];
        captureDescriptor.captureObject = self.device;

        NSError* error;
        if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error]) {
            NSLog(@"Failed to start capture, error %@", error);
        }
    }
}

- (void)stopCapture {
    if (@available(iOS 13.0, *)) {
        MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
        [captureManager stopCapture];
    }
}

@end
