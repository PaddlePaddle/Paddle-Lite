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
#include "lite/utils/cp_logging.h"
#include "lite/kernels/metal/image_op/fetch_image_compute.h"

// debug macro
//#define METAL_DEBUG_GPU_CAPTURE
//#define METAL_DEBUG_ONE_COMMANDBUFFER

extern NSString* cString2NSString(std::string cStr) {
    return [NSString stringWithCString:cStr.c_str() encoding:[NSString defaultCStringEncoding]];
}

@interface MetalContextImp () {
    std::vector<paddle::lite::kernels::metal::FetchImageCompute *> _fetch_vector;
}
@property (strong, nonatomic) id<MTLDevice> device;
@property (strong, nonatomic) id<MTLLibrary> library;
@property (strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property (strong, nonatomic) id<MTLCommandBuffer> commandBuffer;
@property (strong, nonatomic) NSMutableArray<id<MTLCommandBuffer>>* waitings;
@property (strong, nonatomic) NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* caches;
@end

@implementation MetalContextImp

#pragma mark - external

- (instancetype)init {
    self = [super init];
    if (self) {
        _fetch_vector = {};
        _waitings = [NSMutableArray array];
        _caches = [NSMutableDictionary dictionary];
        _device = MTLCreateSystemDefaultDevice();
        _commandQueue = [_device newCommandQueue];
#if defined (METAL_DEBUG_GPU_CAPTURE)
        // At least one command buffer must be created and
        //committed within the boundaries of a GPU Capture.
        [self startCapture];
#endif
        _commandBuffer = [_commandQueue commandBuffer];
    }
    return self;
}

- (void)dealloc {
    _fetch_vector.clear();
}

- (void)setMetalPath:(std::string)path {
    NSString *pathStr = cString2NSString(path);
    if (pathStr) {
        _library = [self.device newLibraryWithFile:pathStr error:NULL];
    }
    if (nil == _library) {
        LOG(INFO) << "Can't load metallib: "
                  << [pathStr cStringUsingEncoding:NSUTF8StringEncoding];
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

- (id<MTLHeap>)newHeapForTexDesc:(MTLTextureDescriptor*)desc {
    NSInteger size = [_device heapTextureSizeAndAlignWithDescriptor:desc].size;
    MTLHeapDescriptor* heapDesc = [[MTLHeapDescriptor alloc] init];
    heapDesc.size = size;
    heapDesc.storageMode = MTLStorageModeShared;
    return [_device newHeapWithDescriptor:heapDesc];
}

- (bool)isNeedNewHeap:(id<MTLHeap>)heap texDesc:(MTLTextureDescriptor*)desc {
    NSInteger hSize = heap.size;
    NSInteger tSize = [_device heapTextureSizeAndAlignWithDescriptor:desc].size;
    if (hSize >= tSize) {
        return false;
    } else {
        return true;
    }
}

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor*)desc heap:(id<MTLHeap>)heap {
    id<MTLTexture> image = [heap newTextureWithDescriptor:desc];
    [image makeAliasable];
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
#if defined (METAL_DEBUG_ONE_COMMANDBUFFER)

#else
    [_commandBuffer commit];
    [_waitings addObject:_commandBuffer];
    _commandBuffer = [_commandQueue commandBuffer];
#endif
}

// mps
- (id<MTLCommandBuffer>)commandBuffer {
#if defined (METAL_DEBUG_ONE_COMMANDBUFFER)
    return _commandBuffer;
#else
    id<MTLCommandBuffer> result = [_commandQueue commandBuffer];
    assert(nil != result);
    return result;
#endif
}

// mps
- (void)commit:(id<MTLCommandBuffer>)cmdBuf {
    assert(nil != cmdBuf);
#if defined (METAL_DEBUG_ONE_COMMANDBUFFER)

#else
    [cmdBuf commit];
    [_waitings addObject:cmdBuf];
#endif
}

- (void)waitAllCompleted {
#if defined (METAL_DEBUG_ONE_COMMANDBUFFER)
    [_commandBuffer commit];
#if defined (METAL_DEBUG_GPU_CAPTURE)
    [self stopCapture];
#endif
    [_commandBuffer waitUntilCompleted];
    _commandBuffer = [_commandQueue commandBuffer];

#else
    
#if defined (METAL_DEBUG_GPU_CAPTURE)
    [self stopCapture];
#endif
    for (id<MTLCommandBuffer> buffer in _waitings) {
        if (buffer.status >= MTLCommandBufferStatusCompleted) {
            continue;
        }
        [buffer waitUntilCompleted];
        if (buffer.error) {
            LOG(INFO) << "[METAL]: " << buffer.error.localizedDescription.UTF8String;
        }
    }
    [_waitings removeAllObjects];
#endif
}

#pragma mark c++ external

- (void)add_fetch_kernel_ptr:(void *)ptr {
    paddle::lite::kernels::metal::FetchImageCompute *fetch =
        static_cast<paddle::lite::kernels::metal::FetchImageCompute *>(ptr);
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
    [encoder setComputePipelineState:pipline];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
}

- (void)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                pipline:(id<MTLComputePipelineState>)pipline
        threadsPerGroup:(MTLSize)threadsPerGroup
                 groups:(MTLSize)groups {
    [encoder setComputePipelineState:pipline];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
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
    MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
    MTLCaptureDescriptor* captureDescriptor = [[MTLCaptureDescriptor alloc] init];
    captureDescriptor.captureObject = self.device;

    NSError *error;
    if (![captureManager startCaptureWithDescriptor: captureDescriptor error:&error]) {
        NSLog(@"Failed to start capture, error %@", error);
    }
}

- (void)stopCapture {
    MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
    [captureManager stopCapture];
}

@end
