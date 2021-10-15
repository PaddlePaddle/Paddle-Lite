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
// #define METAL_DEBUG_GPU_CAPTURE

extern NSString* cString2NSString(std::string cStr) {
    return [NSString stringWithCString:cStr.c_str() encoding:[NSString defaultCStringEncoding]];
}

@interface MetalContextImp () {
    bool use_one_cmdbuf_;
    bool use_memory_reuse_;
    std::vector<paddle::lite_metal::kernels::metal::FetchImageCompute *> _fetch_vector;
}
@property (strong, nonatomic) id<MTLDevice> device;
@property (strong, nonatomic) NSString* libraryPath;
@property (strong, nonatomic) id<MTLLibrary> library;
@property (strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property (strong, nonatomic) id<MTLCommandBuffer> commandBuffer;
@property (strong, nonatomic) NSMutableArray<id<MTLCommandBuffer>>* waitings;
@property (strong, nonatomic) NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* caches;
@property (strong, nonatomic) NSMutableDictionary<NSString*, NSArray*>* resizeInputs;
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
        _resizeInputs = [NSMutableDictionary dictionaryWithCapacity:3];
        _memoryReuseHeaps = [NSMutableDictionary dictionaryWithCapacity:3];
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
        self.libraryPath = pathStr;
        self.library = [self.device newLibraryWithFile:pathStr error:NULL];
    }
    if (nil == _library) {
        LOG(INFO) << "Can't load metallib: "
                  << [pathStr cStringUsingEncoding:NSUTF8StringEncoding];
    }
}

- (void)setMetalDevice:(void *)device {
    if (!device) {
        return;
    }
    id<MTLDevice> mtl = (__bridge id<MTLDevice>)device;
    if (mtl) {
        self.device = mtl;
        self.commandQueue = [_device newCommandQueue];
#if defined (METAL_DEBUG_GPU_CAPTURE)
        [self startCapture];
#endif
        self.commandBuffer = [self.commandQueue commandBuffer];
    }
    if (self.libraryPath) {
        [self setMetalPath:self.libraryPath.UTF8String];
    }
}

#pragma mark data

- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size access:(paddle::lite_metal::METAL_ACCESS_FLAG)access {
    return [_device newBufferWithLength:size options:[self optionForAccess:access]];
}

- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size
                           bytes:(void*)bytes
                          access:(paddle::lite_metal::METAL_ACCESS_FLAG)access {
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
    if(use_one_cmdbuf_){
        
    } else {
        [_commandBuffer commit];
        [_waitings addObject:_commandBuffer];
        _commandBuffer = [_commandQueue commandBuffer];
    }
}

// mps
- (id<MTLCommandBuffer>)commandBuffer {
    if(use_one_cmdbuf_){
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
    if(use_one_cmdbuf_){

    } else {
        [cmdBuf commit];
        [_waitings addObject:cmdBuf];
    }
}

- (void)waitAllCompleted {
    if(use_one_cmdbuf_){
        [_commandBuffer commit];
#if defined (METAL_DEBUG_GPU_CAPTURE)
        [self stopCapture];
#endif
        [_commandBuffer waitUntilCompleted];
        _commandBuffer = [_commandQueue commandBuffer];
    } else {
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
    }
}

#pragma mark c++ external

- (void)add_fetch_kernel_ptr:(void *)ptr {
    paddle::lite_metal::kernels::metal::FetchImageCompute *fetch =
        static_cast<paddle::lite_metal::kernels::metal::FetchImageCompute *>(ptr);
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

#pragma mark pre-process

- (void)resizeInput:(int64_t)index texture:(void *)texture dims:(std::vector<int64_t>&)dims  {
    id<MTLTexture> inTexture = (__bridge id<MTLTexture>)texture;
    if (inTexture) {
        NSMutableArray *dimsAry = [NSMutableArray arrayWithCapacity:3];
        for (int i = 0; i < dims.size(); i++) {
            [dimsAry addObject:@(dims[i])];
        }
        [self.resizeInputs setObject:@[inTexture, dimsAry] forKey:@(index).stringValue];
    }
}

- (NSArray *)getResizeInput:(int64_t)index {
    if (index < self.resizeInputs.allKeys.count) {
        NSArray *resizeAry = [self.resizeInputs objectForKey:@(index).stringValue];
        if (resizeAry) {
            return resizeAry;
        }
    }
    return nil;
}

- (MPSImageLanczosScale *)lanczosScalePtrCreate {
    MPSImageLanczosScale *lanczos = [[MPSImageLanczosScale alloc] initWithDevice:self.device];
    return lanczos;
}

- (id<MTLTexture>)lanczosTextureCreate:(NSArray *)dims {
    MTLTextureDescriptor *textureDesc = [[MTLTextureDescriptor alloc] init];
    textureDesc.textureType = MTLTextureType2D;
    textureDesc.width = [dims[3] intValue];
    textureDesc.height = [dims[2] intValue];
    textureDesc.depth = ([dims[1] intValue] + 3) / 4;
    textureDesc.pixelFormat = MTLPixelFormatRGBA16Float;
    textureDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;;
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
        return nil;
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

- (bool)isNewHeapWithDescriptor:(MTLTextureDescriptor*)desc heap:(id<MTLHeap>)heap API_AVAILABLE(ios(10.0)) {
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

- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor*)desc heap:(id<MTLHeap>)heap API_AVAILABLE(ios(10.0)) {
    if (@available(iOS 10.0, *)) {
        id<MTLTexture> image = [heap newTextureWithDescriptor:desc];
        [image makeAliasable];
        assert(nil != image);
        return image;
    }
    return nil;
}

#pragma mark - internal

- (MTLResourceOptions)optionForAccess:(paddle::lite_metal::METAL_ACCESS_FLAG)access {
    if (access == paddle::lite_metal::METAL_ACCESS_FLAG::CPUWriteOnly) {
        return MTLResourceOptionCPUCacheModeWriteCombined;
    } else if (access == paddle::lite_metal::METAL_ACCESS_FLAG::CPUTransparent) {
        if (@available(iOS 9.0, *)) {
            return MTLResourceStorageModePrivate;
        } else {
            return MTLResourceOptionCPUCacheModeDefault;
        }
    } else if (access == paddle::lite_metal::METAL_ACCESS_FLAG::CPUShared) {
        return MTLStorageModeShared;
    } else {  // access == paddle::lite_metal::METAL_ACCESS_FLAG::CPUReadWrite
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

        NSError *error;
        if (![captureManager startCaptureWithDescriptor: captureDescriptor error:&error]) {
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
