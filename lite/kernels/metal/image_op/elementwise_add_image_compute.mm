// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/metal/image_op/elementwise_add_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void ElementwiseAddImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.Out->dims();
    auto input_dims = param.X->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
    input_buffer_x_ = param.X->data<MetalHalf, MetalImage>();
    input_buffer_y_ = param.Y->data<MetalHalf, MetalImage>();
#endif

    // use MPS or not
    bool should_use_mps = false;
    if (@available(iOS 10.0, *)) {
        if (metal_context_->use_mps()) {
            should_use_mps = true;
        }
    }
    // X Y same dims
    if ((input_buffer_x_->dim_ == input_buffer_y_->dim_) &&
        (input_buffer_x_->transpose_ == input_buffer_y_->transpose_)) {
    } else {
        should_use_mps = false;
    }
// X Y output
#ifdef LITE_WITH_METAL_FULL
    if ([input_buffer_x_->image() pixelFormat] == MTLPixelFormatRGBA32Float &&
        [input_buffer_y_->image() pixelFormat] == MTLPixelFormatRGBA32Float &&
        [output_buffer_->image() pixelFormat] == MTLPixelFormatRGBA32Float) {
    } else {
        should_use_mps = false;
    }
#else
    if ([input_buffer_x_->image() pixelFormat] == MTLPixelFormatRGBA16Float &&
        [input_buffer_y_->image() pixelFormat] == MTLPixelFormatRGBA16Float &&
        [output_buffer_->image() pixelFormat] == MTLPixelFormatRGBA16Float) {
    } else {
        should_use_mps = false;
    }
#endif

    use_mps_ = should_use_mps;
    if (use_mps_) {
        setup_with_mps();
    } else {
        setup_without_mps();
    }
}

void ElementwiseAddImageCompute::Run() {
    if (use_mps_) {
        run_with_mps();
    } else {
        run_without_mps();
    }
}

#pragma mark - SELF

void ElementwiseAddImageCompute::run_without_mps() {
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:(input_buffer_x_->image()) atIndex:(0)];
    [encoder setTexture:(input_buffer_y_->image()) atIndex:(1)];
    [encoder setTexture:(output_buffer_->image()) atIndex:(2)];
    [encoder setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void ElementwiseAddImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();
    auto output_dims = param.Out->dims();
    auto input_dims = param.X->dims();

    std::vector<int> xdim, ydim;
    for (int i = 0; i < 4; i++) {
        xdim.push_back((int)input_buffer_x_->dim_[i]);
        ydim.push_back((int)input_buffer_y_->dim_[i]);
    }

    auto axis = param.axis;
    int params_axis = 0;
    if (axis == -1) {
        params_axis = 4 - (int)(input_buffer_y_->tensor_dim_.size());
    } else {
        params_axis = 4 - (int)(input_buffer_x_->tensor_dim_.size()) + axis;
    }
    int params_fast = 0;
    if ((input_buffer_x_->dim_ == input_buffer_y_->dim_) &&
        (input_buffer_x_->transpose_ == input_buffer_y_->transpose_)) {
        params_fast = 1;
    }

    int add_by_channel = 0;
    if (input_buffer_y_->tensor_dim_.size() == 1 &&
        (axis == 1 ||
            (axis == -1 &&
                input_buffer_y_->tensor_dim_[0] == input_buffer_x_->dim_[3]))) {
        add_by_channel = 1;
    }
    if (add_by_channel == 1 || params_fast == 1) {
    } else {
        LOG(FATAL) << "elementwise_add: add only support by channel";
    }

    ElementwiseAddMetalParam element_params = {params_fast,
        add_by_channel,
        params_axis,
        (int)input_buffer_y_->tensor_dim_.size(),
        {xdim[0], xdim[1], xdim[2], xdim[3]},
        {input_buffer_x_->transpose_[0],
            input_buffer_x_->transpose_[1],
            input_buffer_x_->transpose_[2],
            input_buffer_x_->transpose_[3]},
        {ydim[0], ydim[1], ydim[2], ydim[3]},
        {input_buffer_y_->transpose_[0],
            input_buffer_y_->transpose_[1],
            input_buffer_y_->transpose_[2],
            input_buffer_y_->transpose_[3]}};

    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(element_params), &element_params);

    function_name_ = "elementwise_add";

    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

#pragma mark - MPS

void ElementwiseAddImageCompute::run_with_mps() {
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    auto cmdbuf = [backend commandBuffer];
    if (mps_add_op_) {
        [((__bridge MPSCNNAdd*)mps_add_op_)
            encodeToCommandBuffer:cmdbuf
                     primaryImage:(__bridge MPSImage*)mps_input_image_
                   secondaryImage:(__bridge MPSImage*)mps_input_image_y_
                 destinationImage:(__bridge MPSImage*)mps_output_image_];
    }
    [backend commit:cmdbuf];
}

void ElementwiseAddImageCompute::setup_with_mps() {
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    //
    mps_add_op_ = (__bridge_retained void*)[[MPSCNNAdd alloc] initWithDevice:backend.device];
    // MPS算子输入输出
    auto input_x_c = MAX(4, static_cast<int>(input_buffer_x_->tensor_dim_[1]));
    auto input_y_c = MAX(4, static_cast<int>(input_buffer_y_->tensor_dim_[1]));
    auto output_c = MAX(4, static_cast<int>(output_buffer_->tensor_dim_[1]));
    mps_input_image_ =
        (__bridge_retained void*)[[MPSImage alloc] initWithTexture:input_buffer_x_->image()
                                                   featureChannels:input_x_c];
    mps_input_image_y_ =
        (__bridge_retained void*)[[MPSImage alloc] initWithTexture:input_buffer_y_->image()
                                                   featureChannels:input_y_c];
    mps_output_image_ =
        (__bridge_retained void*)[[MPSImage alloc] initWithTexture:output_buffer_->image()
                                                   featureChannels:output_c];
}

ElementwiseAddImageCompute::~ElementwiseAddImageCompute() {
    if (mps_add_op_) {
        CFRelease(mps_add_op_);
        mps_add_op_ = nullptr;
    }
    if (mps_input_image_) {
        CFRelease(mps_input_image_);
        mps_input_image_ = nullptr;
    }
    if (mps_input_image_y_) {
        CFRelease(mps_input_image_y_);
        mps_input_image_y_ = nullptr;
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

REGISTER_LITE_KERNEL(elementwise_add,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseAddImageCompute,
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
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_add,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseAddImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
