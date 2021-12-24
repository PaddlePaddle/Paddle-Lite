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

#ifndef LITE_BACKENDS_METAL_METAL_CONTEXT_IMP_H_
#define LITE_BACKENDS_METAL_METAL_CONTEXT_IMP_H_

#include "lite/backends/metal/metal_common.h"
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string>

extern NSString* cString2NSString(std::string cStr);

@interface MetalContextImp : NSObject
@property (strong, nonatomic, readonly) id<MTLDevice> device;

- (void)setMetalPath:(std::string)path;
- (void)setMetalDevice:(void*)device;

- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size access:(paddle::lite::METAL_ACCESS_FLAG)access;
- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size
                           bytes:(void*)bytes
                          access:(paddle::lite::METAL_ACCESS_FLAG)access;
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor*)desc;

// for MPS
- (id<MTLCommandBuffer>)commandBuffer;
- (void)commit:(id<MTLCommandBuffer>)cmdBuf;
// for custom kernel
- (id<MTLComputeCommandEncoder>)commandEncoder;
- (id<MTLComputePipelineState>)pipline:(std::string)name;

- (void)commit;
- (void)waitAllCompleted;

- (void)fetch_data_from_gpu;
- (void)add_fetch_kernel_ptr:(void*)ptr;

- (void)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                pipline:(id<MTLComputePipelineState>)pipline
             outTexture:(id<MTLTexture>)outTexture;

- (void)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                pipline:(id<MTLComputePipelineState>)pipline
             outTexture:(id<MTLTexture>)outTexture
              quadruple:(bool)quadruple;

- (void)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                pipline:(id<MTLComputePipelineState>)pipline
        threadsPerGroup:(MTLSize)threadsPerGroup
                 groups:(MTLSize)groups;

- (void)dispatchEncoder:(id<MTLComputeCommandEncoder>)encoder
                pipline:(id<MTLComputePipelineState>)pipline
           threadsShape:(NSArray<NSNumber*>*)threadsShape;

// pre-process
- (MPSImageLanczosScale*)lanczosScalePtrCreate;
- (id<MTLTexture>)lanczosTextureCreate:(NSArray*)dims;

// memory reuse
- (void)set_use_memory_reuse:(bool)flag;
- (void)setHeap:(id<MTLHeap>)heap key:(std::string)ptr API_AVAILABLE(ios(10.0));
- (id<MTLHeap>)getHeap:(std::string)ptr API_AVAILABLE(ios(10.0));
- (id<MTLHeap>)newHeapWithDescriptor:(MTLTextureDescriptor*)desc API_AVAILABLE(ios(10.0));
- (bool)isNewHeapWithDescriptor:(MTLTextureDescriptor*)desc
                           heap:(id<MTLHeap>)heap API_AVAILABLE(ios(10.0));
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor*)desc
                                      heap:(id<MTLHeap>)heap API_AVAILABLE(ios(10.0));

@end

#endif  // LITE_BACKENDS_METAL_METAL_CONTEXT_IMP_H_
