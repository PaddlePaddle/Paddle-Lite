//
//  metal_context_oc.h
//  PaddleLiteiOS
//
//  Created by hxwc on 2021/3/22.
//

#ifndef METAL_CONTEXT_OC_H
#define METAL_CONTEXT_OC_H

#include "lite/backends/metal/metal_common.h"
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string>

extern NSString* cString2NSString(std::string cStr);

@interface MetalContextImp : NSObject
@property (strong, nonatomic, readonly) id<MTLDevice> device;

- (void)setMetalPath:(std::string)path;

- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size access:(paddle::lite::METAL_ACCESS_FLAG)access;
- (id<MTLBuffer>)newDeviceBuffer:(NSUInteger)size
                           bytes:(void*)bytes
                          access:(paddle::lite::METAL_ACCESS_FLAG)access;
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor*)desc;

- (id<MTLHeap>)newHeapForTexDesc:(MTLTextureDescriptor*)desc;
- (bool)isNeedNewHeap:(id<MTLHeap>)heap texDesc:(MTLTextureDescriptor*)desc;
- (id<MTLTexture>)newTextureWithDescriptor:(MTLTextureDescriptor*)desc heap:(id<MTLHeap>)heap;

// MPS使用
- (id<MTLCommandBuffer>)commandBuffer;
- (void)commit:(id<MTLCommandBuffer>)cmdBuf;
// SELF算子使用
- (id<MTLComputeCommandEncoder>)commandEncoder;
- (id<MTLComputePipelineState>)pipline:(std::string)name;

- (void)commit;
- (void)waitAllCompleted;
- (void)waitUntilCompleted;

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

@end

#endif
