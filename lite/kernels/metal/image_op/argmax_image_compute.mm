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
#include "lite/kernels/metal/image_op/argmax_image_compute.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void ArgmaxImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.Out->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.X->data<MetalHalf, MetalImage>();
    output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(
        metal_context_, MetalImage::FourDimFrom(output_dims));
#endif

    // use mps or not
    bool should_use_mps = false;
    if (@available(iOS 12.0, *)) {
        if (metal_context_->use_mps()) {
            if (param.Axis == 1) should_use_mps = true;
        }
    }
    use_mps_ = should_use_mps;
    if (use_mps_) {
        setup_with_mps();
    } else {
        setup_without_mps();
    }
}

void ArgmaxImageCompute::Run() {
    @autoreleasepool {
        if (use_mps_) {
            run_with_mps();
        } else {
            run_without_mps();
        }
    }
}

#pragma mark - SELF
void ArgmaxImageCompute::run_without_mps() {
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:(input_buffer_->image()) atIndex:(0)];
    [encoder setTexture:(output_buffer_->image()) atIndex:(1)];
    [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void ArgmaxImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();
    auto irank = input_buffer_->tensor_dim_.size();
    auto orank = output_buffer_->tensor_dim_.size();

    // axis
    if (irank == 4 && param.Axis == 1) {
        function_name_ = "arg_max_c";
    } else if (irank == 4 && param.Axis == 2) {
        function_name_ = "arg_max_h";
    } else if (irank == 4 && param.Axis == 3) {
        function_name_ = "arg_max_w";
    } else {
        LOG(FATAL) << "This input is not supported by arg_max";
    }

    ArgMetalParam metal_params{(int)orank};
    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(metal_params), &metal_params);

    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

#pragma mark - MPS

void ArgmaxImageCompute::run_with_mps() {
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    auto cmdbuf = [backend commandBuffer];
    if (mps_op_) {
        if (@available(iOS 12.0, *)) {
            [((__bridge MPSNNReduceFeatureChannelsArgumentMax*)mps_op_)
                encodeToCommandBuffer:cmdbuf
                          sourceImage:(__bridge MPSImage*)mps_input_image_
                     destinationImage:(__bridge MPSImage*)mps_output_image_];
        }
    }
    [backend commit:cmdbuf];
    // attention: mps output c=1
    {
        std::string function = "tex2d_c1_to_c4";
        id<MTLComputePipelineState> pipline = [backend pipline:function];

        auto outTexture = output_buffer_->image();
        auto backend = (__bridge MetalContextImp*)metal_context_->backend();

        auto encoder = [backend commandEncoder];
        [encoder setTexture:([(__bridge MPSImage*)mps_output_image_ texture]) atIndex:(0)];
        [encoder setTexture:(output_buffer_->image()) atIndex:(1)];

        [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
        [backend commit];
    }
}

void ArgmaxImageCompute::setup_with_mps() {
    const auto& param = this->Param<param_t>();
    auto irank = input_buffer_->tensor_dim_.size();
    auto orank = output_buffer_->tensor_dim_.size();
    // axis
    if (param.Axis == 1 && irank == 4 && orank == 4) {
    } else {
        LOG(FATAL) << "mps_arg_max: max only support by channel";
    }

    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    if (@available(iOS 12.0, *)) {
        mps_op_ = (__bridge_retained void*)[[MPSNNReduceFeatureChannelsArgumentMax alloc]
            initWithDevice:backend.device];
        // MPS input and output
        auto input_c = fmax(4, static_cast<int>(input_buffer_->tensor_dim_[1]));
        mps_input_image_ =
            (__bridge_retained void*)[[MPSImage alloc] initWithTexture:input_buffer_->image()
                                                       featureChannels:input_c];
        // mps texture featureChannels must be 1
        auto output_dims = param.Out->dims();
        auto dim = MetalImage::FourDimFrom(output_dims);
        auto metal_image = new MetalImage(metal_context_,
            dim,
            {0, 2, 3, 1},
            METAL_PRECISION_TYPE::HALF,
            METAL_ACCESS_FLAG::CPUReadWrite,
            true);
        metal_image->initImage(metal_context_);
        mps_output_image_ = (__bridge_retained void*)[[MPSImage alloc]
            initWithTexture:metal_image->image()
            featureChannels:metal_image->channels_per_pixel_];
        free(metal_image);
    }
}

ArgmaxImageCompute::~ArgmaxImageCompute() {
    if (mps_op_) {
        CFRelease(mps_op_);
        mps_op_ = nullptr;
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

REGISTER_LITE_KERNEL(arg_max,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ArgmaxImageCompute,
    Int32)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kInt32),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(arg_max,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ArgmaxImageCompute,
    Int64)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kInt64),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
