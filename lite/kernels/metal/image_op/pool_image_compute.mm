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

#include <cmath>

#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/kernels/metal/image_op/pool_image_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void PoolImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.output->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.x->data<MetalHalf, MetalImage>();
    output_buffer_ = param.output->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif

    // use mps or not
    bool should_use_mps = false;
    if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
        if (metal_context_->use_mps()) {
            int input_c = static_cast<int>(input_buffer_->tensor_dim_[1]);
            int output_c = static_cast<int>(output_buffer_->tensor_dim_[1]);
            if (input_c >= 3 && output_c >= 3) {
                should_use_mps = true;
            }
        }
    }
    if (param.global_pooling) {
    }
    use_mps_ = should_use_mps;
    if (use_mps_) {
        setup_with_mps();
    } else {
        setup_without_mps();
    }
}

void PoolImageCompute::Run() {
    @autoreleasepool {
        if (use_mps_) {
            run_with_mps();
        } else {
            run_without_mps();
        }
    }
}

#pragma mark - SELF
void PoolImageCompute::run_without_mps() {
    const auto& param = this->Param<param_t>();
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    if (param.global_pooling) {
        [encoder setTexture:(input_buffer_->image()) atIndex:(0)];
        [encoder setTexture:(output_buffer_->image()) atIndex:(1)];

        // according to 'global_pooling' set 'threadgroup'
        // A14: maybe use SIMD reduction instructions
        auto inTexture = input_buffer_->image();
        NSUInteger slices = (outTexture.arrayLength * 4 + 3) / 4;
        NSUInteger width = 0, height = 0, groupWidth = 0, groupHeight = 0;
        width = MIN(256, pipline.threadExecutionWidth);
        width = MIN(width, inTexture.width);
        height = MIN(256, pipline.maxTotalThreadsPerThreadgroup) / width;
        height = MIN(height, inTexture.height);
        groupWidth = 1;
        groupHeight = 1;
        MTLSize threadsPerGroup = MTLSize{.width = width, .height = height, .depth = 1};
        MTLSize groups = MTLSize{.width = groupWidth, .height = groupHeight, .depth = slices};
        [backend dispatchEncoder:encoder
                         pipline:pipline
                 threadsPerGroup:threadsPerGroup
                          groups:groups];
    } else {
        [encoder setTexture:(input_buffer_->image()) atIndex:(0)];
        [encoder setTexture:(output_buffer_->image()) atIndex:(1)];
        [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];

        [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    }
    [backend commit];
}

void PoolImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();

    int pool_type = 0;
    if (param.pooling_type == "max")
        pool_type = 0;
    else if (param.pooling_type == "avg")
        pool_type = 1;
    else {
        LOG(FATAL) << "pool: no such pooling type\n";
    }
    if (param.global_pooling) {
        if (pool_type == 1) {
            // global_pooling only support 'avg'
            function_name_ = "global_pool";
        } else {
            LOG(FATAL) << "pool: global_pooling no such pooling type\n";
        }
    } else {
        auto kw = param.ksize[1];
        auto kh = param.ksize[0];
        auto sw = param.strides[1];
        auto sh = param.strides[0];
        auto pw = (*param.paddings)[2];
        auto ph = (*param.paddings)[0];

        PoolMetalParam pool_params{kw, kh, sw, sh, pw, ph, pool_type, param.exclusive};

        params_buffer_ =
            std::make_shared<MetalBuffer>(metal_context_, sizeof(pool_params), &pool_params);

        function_name_ = "pool";
    }
    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

#pragma mark - MPS

void PoolImageCompute::run_with_mps() {
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    auto cmdbuf = [backend commandBuffer];
    if (mps_pool_op_) {
        if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
            [((__bridge MPSCNNPooling*)mps_pool_op_)
                encodeToCommandBuffer:cmdbuf
                          sourceImage:(__bridge MPSImage*)mps_input_image_
                     destinationImage:(__bridge MPSImage*)mps_output_image_];
        }
    }
    [backend commit:cmdbuf];
}

void PoolImageCompute::setup_with_mps() {
    const auto& param = this->Param<param_t>();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto kw = param.ksize[1];
    auto kh = param.ksize[0];
    auto sw = param.strides[1];
    auto sh = param.strides[0];
    auto pw = (*param.paddings)[2];
    auto ph = (*param.paddings)[0];

    if (param.global_pooling) {
        auto input_dims = param.x->dims();
        kw = (int)input_dims[3];
        kh = (int)input_dims[2];
        pw = 0;
        ph = 0;
    }
    int offsetX = static_cast<int>(((int)(kw - 1) + 1) / 2 - pw);
    int offsetY = static_cast<int>(((int)(kh - 1) + 1) / 2 - ph);

    if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
        if (param.pooling_type == "max") {
            mps_pool_op_ =
                (__bridge_retained void*)[[MPSCNNPoolingMax alloc] initWithDevice:backend.device
                                                                      kernelWidth:kw
                                                                     kernelHeight:kh
                                                                  strideInPixelsX:sw
                                                                  strideInPixelsY:sh];
            ((__bridge MPSCNNPoolingMax*)mps_pool_op_).offset =
                MPSOffset{.x = offsetX, .y = offsetY};
            ((__bridge MPSCNNPoolingMax*)mps_pool_op_).edgeMode = MPSImageEdgeModeZero;
        } else if (param.pooling_type == "avg") {
            mps_pool_op_ = (__bridge_retained void*)[[MPSCNNPoolingAverage alloc]
                 initWithDevice:backend.device
                    kernelWidth:input_buffer_->image().width
                   kernelHeight:input_buffer_->image().height
                strideInPixelsX:input_buffer_->image().width
                strideInPixelsY:input_buffer_->image().height];
            ((__bridge MPSCNNPoolingAverage*)mps_pool_op_).offset =
                MPSOffset{.x = static_cast<NSInteger>(input_buffer_->image().width / 2),
                    .y = static_cast<NSInteger>(input_buffer_->image().height / 2)};
            ((__bridge MPSCNNPoolingAverage*)mps_pool_op_).edgeMode = MPSImageEdgeModeZero;
        }
        // MPS input and output
        auto input_c = static_cast<int>(input_buffer_->tensor_dim_[1]);
        auto output_c = static_cast<int>(output_buffer_->tensor_dim_[1]);
        mps_input_image_ =
            (__bridge_retained void*)[[MPSImage alloc] initWithTexture:input_buffer_->image()
                                                       featureChannels:input_c];
        mps_output_image_ =
            (__bridge_retained void*)[[MPSImage alloc] initWithTexture:output_buffer_->image()
                                                       featureChannels:output_c];
    }
}

PoolImageCompute::~PoolImageCompute() {
    if (mps_pool_op_) {
        CFRelease(mps_pool_op_);
        mps_pool_op_ = nullptr;
    }
    if (mps_input_image_) {
        CFRelease(mps_input_image_);
        mps_input_image_ = nullptr;
    }
    if (mps_output_image_) {
        CFRelease(mps_output_image_);
        mps_output_image_ = nullptr;
    }
    TargetWrapperMetal::FreeImage(output_buffer_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#pragma mark -

REGISTER_LITE_KERNEL(pool2d,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::PoolImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(pool2d,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::PoolImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
