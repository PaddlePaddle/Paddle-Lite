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

#include "lite/kernels/metal/image_op/compare_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void CompareImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.Out->dims();
    auto input_dims = param.X->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_x_ = param.X->data<MetalHalf, MetalImage>();
    input_buffer_y_ = param.Y->data<MetalHalf, MetalImage>();
    output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif

    // use mps or not
    bool should_use_mps = false;
    if (@available(iOS 12.1, *)) {
        if (metal_context_->use_mps()) {
            if (input_buffer_x_->tensor_dim_[0] == 1) should_use_mps = true;
        }
    }

    use_mps_ = should_use_mps;
    if (use_mps_) {
        setup_with_mps();
    } else {
        setup_without_mps();
    }
}

void CompareImageCompute::Run() {
    @autoreleasepool {
        if (use_mps_) {
            run_with_mps();
        } else {
            run_without_mps();
        }
    }
}

#pragma mark - SELF

void CompareImageCompute::run_without_mps() {
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:input_buffer_x_->image() atIndex:(0)];
    [encoder setTexture:input_buffer_y_->image() atIndex:(1)];
    [encoder setTexture:output_buffer_->image() atIndex:(2)];
    [encoder setBuffer:params_buffer_->buffer() offset:(0) atIndex:(0)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void CompareImageCompute::setup_without_mps() {
    auto valid = true;
    if (input_buffer_x_->tensor_dim_.size() == input_buffer_y_->tensor_dim_.size()) {
        valid = true;
        for (int i = 0; i < input_buffer_x_->tensor_dim_.size(); i++) {
            if (input_buffer_x_->tensor_dim_[i] != input_buffer_y_->tensor_dim_[i]) {
                valid = false;
                break;
            }
        }
    }
    if (!valid) {
        LOG(FATAL) << "compare: only supports : same shapes";
    }

    // Equal = 0, NotEqual = 1, LessThan = 2, LessEqual = 3, GreaterThan = 4, GreaterEqual = 5,
    int compareType = 0;
    CompareMetalParam metal_params = {compareType};
    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(metal_params), &metal_params);

    // pipline
    function_name_ = "compare";
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

#pragma mark - MPS

void CompareImageCompute::run_with_mps() {
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    auto cmdbuf = [backend commandBuffer];
    if (mps_op_) {
        if (@available(iOS 12.1, *)) {
            [((__bridge MPSNNCompare*)mps_op_)
                encodeToCommandBuffer:cmdbuf
                         primaryImage:(__bridge MPSImage*)mps_input_x_image_
                       secondaryImage:(__bridge MPSImage*)mps_input_y_image_
                     destinationImage:(__bridge MPSImage*)mps_output_image_];
        }
    }
    [backend commit:cmdbuf];
}

void CompareImageCompute::setup_with_mps() {
    auto xrank = input_buffer_x_->tensor_dim_.size();
    auto yrank = input_buffer_y_->tensor_dim_.size();
    // axis
    if (xrank == 4 && yrank == 4) {
    } else {
        LOG(FATAL) << "mps_compare: max only support by channel";
    }

    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    if (@available(iOS 12.1, *)) {
        mps_op_ = (__bridge_retained void*)[[MPSNNCompare alloc] initWithDevice:backend.device];
        [(__bridge MPSNNCompare*)mps_op_ setComparisonType:MPSNNComparisonTypeEqual];
        // MPS input and output
        auto input_x_c = MAX(4, static_cast<int>(input_buffer_x_->tensor_dim_[1]));
        auto input_y_c = MAX(4, static_cast<int>(input_buffer_y_->tensor_dim_[1]));
        auto output_c = MAX(4, static_cast<int>(output_buffer_->tensor_dim_[1]));
        mps_input_x_image_ =
            (__bridge_retained void*)[[MPSImage alloc] initWithTexture:input_buffer_x_->image()
                                                       featureChannels:input_x_c];
        mps_input_y_image_ =
            (__bridge_retained void*)[[MPSImage alloc] initWithTexture:input_buffer_y_->image()
                                                       featureChannels:input_y_c];
        mps_output_image_ =
            (__bridge_retained void*)[[MPSImage alloc] initWithTexture:output_buffer_->image()
                                                       featureChannels:output_c];
    }
}

CompareImageCompute::~CompareImageCompute() {
    if (mps_op_) {
        CFRelease(mps_op_);
        mps_op_ = nullptr;
    }
    if (mps_input_x_image_) {
        CFRelease(mps_input_x_image_);
        mps_input_x_image_ = nullptr;
    }
    if (mps_input_y_image_) {
        CFRelease(mps_input_y_image_);
        mps_input_y_image_ = nullptr;
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

REGISTER_LITE_KERNEL(equal,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::CompareImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kBool), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(equal,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::CompareImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kBool), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
