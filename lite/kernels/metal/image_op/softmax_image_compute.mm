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

#include "lite/kernels/metal/image_op/softmax_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void SoftmaxImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto input_dims = param.x->dims();
    auto output_dims = param.output->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.x->data<MetalHalf, MetalImage>();
    output_buffer_ = param.output->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif

    auto axis = param.axis;
    if (axis < 0) {
        axis += input_dims.size();
    }
    // whether to use mps
    bool should_use_mps = false;
    if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
        if (metal_context_->use_mps()) {
            int input_c = static_cast<int>(input_buffer_->dim_[3]);
            int output_c = static_cast<int>(output_buffer_->dim_[3]);
            if (input_c >= 3 && output_c >= 3 && input_dims.size() == 4 && axis == 1) {
                should_use_mps = true;
            }
        }
    }
    use_mps_ = should_use_mps;
    if (use_mps_) {
        setup_with_mps();
    } else {
        setup_without_mps();
    }
}

void SoftmaxImageCompute::Run() {
    @autoreleasepool {
        if (use_mps_) {
            run_with_mps();
        } else {
            run_without_mps();
        }
    }
}

#pragma mark - SELF

void SoftmaxImageCompute::run_without_mps() {
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

void SoftmaxImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();
    auto input_dims = param.x->dims();

    if (input_dims.size() != 4 && input_dims.size() != 2) {
        LOG(FATAL) << "only support input with rank(dim)=4 and 2";
        return;
    }

    auto axis = param.axis;
    if (axis < 0) {
        axis += input_dims.size();
    }

    std::string function_name = "softmax";
    if (input_dims.size() == 4) {
        if (axis == 1) {
            function_name = "softmax";
        } else if (axis == 2) {
            function_name = "softmax_h_d3_common";
        } else if (axis == 3) {
            function_name = "softmax_w_d3_common";
        }
    }
    if (input_dims.size() == 2) {
        function_name = "softmax_dim2_common";
    }

    function_name_ = function_name;

    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];

    SoftmaxMetalParam2 metal_param{
        (int)input_buffer_->pad_to_four_dim_[0],
        (int)input_buffer_->pad_to_four_dim_[1],
        (int)input_buffer_->pad_to_four_dim_[2],
        (int)input_buffer_->pad_to_four_dim_[3],
    };
    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(metal_param), &metal_param);
}

#pragma mark - MPS

void SoftmaxImageCompute::run_with_mps() {
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    auto cmdbuf = [backend commandBuffer];
    if (mps_softmax_op_) {
        if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
            [((__bridge MPSCNNSoftMax*)mps_softmax_op_)
                encodeToCommandBuffer:cmdbuf
                          sourceImage:(__bridge MPSImage*)mps_input_image_
                     destinationImage:(__bridge MPSImage*)mps_output_image_];
        }
    }
    [backend commit:cmdbuf];
}

void SoftmaxImageCompute::setup_with_mps() {
    if (@available(iOS 10.0, macOS 10.13, macCatalyst 13.0, *)) {
        auto backend = (__bridge MetalContextImp*)metal_context_->backend();
        //
        mps_softmax_op_ =
            (__bridge_retained void*)[[MPSCNNSoftMax alloc] initWithDevice:backend.device];
        ((__bridge MPSCNNSoftMax*)mps_softmax_op_).edgeMode = MPSImageEdgeModeZero;
        // MPS in and out
        int input_c = static_cast<int>(input_buffer_->dim_[3]);
        int output_c = static_cast<int>(output_buffer_->dim_[3]);
        mps_input_image_ =
            (__bridge_retained void*)[[MPSImage alloc] initWithTexture:input_buffer_->image()
                                                       featureChannels:input_c];
        mps_output_image_ =
            (__bridge_retained void*)[[MPSImage alloc] initWithTexture:output_buffer_->image()
                                                       featureChannels:output_c];
    }
}

SoftmaxImageCompute::~SoftmaxImageCompute() {
    if (mps_softmax_op_) {
        CFRelease(mps_softmax_op_);
        mps_softmax_op_ = nullptr;
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

REGISTER_LITE_KERNEL(softmax,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::SoftmaxImageCompute,
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

REGISTER_LITE_KERNEL(softmax,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::SoftmaxImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
