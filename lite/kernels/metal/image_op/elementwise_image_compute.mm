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

#include "lite/kernels/metal/image_op/elementwise_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

bool InputsValid(const MetalImage* input_x_, const MetalImage* input_y_) {
    auto x_dims = input_x_->dim_;
    auto y_dims = input_y_->dim_;

    // check data layout
    if (input_x_->transpose_ != input_y_->transpose_) return false;
    // check data dims equal
    if (x_dims == y_dims) return true;

    if (x_dims[0] == y_dims[0] && x_dims[3] == y_dims[3]) {
        // Input [N H 1 C]
        if (x_dims[1] == y_dims[1] && (x_dims[2] == 1 || y_dims[2] == 1)) return true;
        // Input [N 1 W C]
        if (x_dims[2] == y_dims[2] && (x_dims[1] == 1 || y_dims[1] == 1)) return true;
        // Input [N 1 1 C]
        if ((x_dims[1] == 1 && x_dims[2] == 1) || (y_dims[1] == 1 && y_dims[2] == 1)) return true;
    }
    return false;
}

void ElementwiseImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();
    init_memory();
    init_for_run();
}

void ElementwiseImageCompute::ReInitWhenNeeded() {
    auto input_dims = elementwise_param_->X->dims();
    if (last_input_dims_ != input_dims) {
        release_memory();
        release_mps_memory();
        init_memory();
        init_for_run();
    }
}

void ElementwiseImageCompute::init_memory() {
    if (!param_.is_type<param_t>()) {
        fuse_flag_ = true;
        elementwise_param_ = param_.get_mutable<operators::FusionElementwiseActivationParam>();
    } else {
        fuse_flag_ = false;
        elementwise_param_ = param_.get_mutable<operators::ElementwiseParam>();
    }
    auto output_dims = elementwise_param_->Out->dims();
    auto input_dims = elementwise_param_->X->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    output_buffer_ =
        elementwise_param_->Out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
    input_buffer_x_ = elementwise_param_->X->data<MetalHalf, MetalImage>();
    input_buffer_y_ = elementwise_param_->Y->data<MetalHalf, MetalImage>();

#endif
    last_input_dims_ = input_dims;
}

void ElementwiseImageCompute::init_for_run() {
    // use MPS or not
    bool should_use_mps = false;
    auto ele_type_ = KernelBase::op_type();

    if (@available(iOS 11.3, macOS 10.13.4, macCatalyst 13.0, *)) {
        if (metal_context_->use_mps()) {
            should_use_mps = true;
        }
    }
    // X Y reasonable dims
    should_use_mps = InputsValid(input_buffer_x_, input_buffer_y_);

// X Y output
#ifdef LITE_WITH_METAL_FULL
#else
    if ([input_buffer_x_->image() pixelFormat] == MTLPixelFormatRGBA16Float &&
        [input_buffer_y_->image() pixelFormat] == MTLPixelFormatRGBA16Float &&
        [output_buffer_->image() pixelFormat] == MTLPixelFormatRGBA16Float) {
    } else {
        should_use_mps = false;
    }
#endif
    if (fuse_flag_) {
        const auto* op_param =
            static_cast<const operators::FusionElementwiseActivationParam*>(elementwise_param_);
        auto act_t = op_param->act_type;
        VLOG(4) << "elementwise_add act: " << act_t;
        if (act_t != "relu") {
            LOG(FATAL) << "Unsupported Activation type: " << act_t << ", support Relu only.";
        }
        should_use_mps = false;
    }

    use_mps_ = should_use_mps;
    if (use_mps_) {
        if (ele_type_ == ("elementwise_add")) {
            setup_with_mps<MPSCNNAdd>();
        } else if (ele_type_ == ("elementwise_div")) {
            setup_with_mps<MPSCNNDivide>();
        } else if (ele_type_ == ("elementwise_mul")) {
            setup_with_mps<MPSCNNMultiply>();
        } else if (ele_type_ == ("elementwise_sub")) {
            setup_with_mps<MPSCNNSubtract>();
        }
    } else {
        if (ele_type_ == ("elementwise_add")) {
            arithmetic_type = 0;
        } else if (ele_type_ == ("elementwise_div")) {
            arithmetic_type = 3;
        } else if (ele_type_ == ("elementwise_mul")) {
            arithmetic_type = 2;
        } else if (ele_type_ == ("elementwise_sub")) {
            arithmetic_type = 1;
        }
        setup_without_mps();
    }
}

void ElementwiseImageCompute::Run() {
    @autoreleasepool {
        auto ele_type_ = KernelBase::op_type();
        if (use_mps_) {
            if (ele_type_ == ("elementwise_add"))
                run_with_mps<MPSCNNAdd>();
            else if (ele_type_ == ("elementwise_div"))
                run_with_mps<MPSCNNDivide>();
            else if (ele_type_ == ("elementwise_mul"))
                run_with_mps<MPSCNNMultiply>();
            else if (ele_type_ == ("elementwise_sub"))
                run_with_mps<MPSCNNSubtract>();
        } else {
            run_without_mps();
        }
    }
}

#pragma mark - SELF

void ElementwiseImageCompute::run_without_mps() {
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
void ElementwiseImageCompute::setup_without_mps() {
    auto x_dims = input_buffer_x_->tensor_dim_;
    auto y_dims = input_buffer_y_->tensor_dim_;
    auto axis = elementwise_param_->axis;
    auto ele_type_ = KernelBase::op_type();

    std::vector<int> xdim, ydim;
    for (int i = 0; i < 4; i++) {
        xdim.push_back((int)input_buffer_x_->dim_[i]);
        ydim.push_back((int)input_buffer_y_->dim_[i]);
    }
    int params_axis = 0;
    int params_fast = 0;
    int by_channel = 0;
    int by_num = 0;
    int by_HW = 0;
    int by_W = 0;

    if (axis == -1) {
        params_axis = 4 - (int)(input_buffer_y_->tensor_dim_.size());
    } else {
        params_axis = 4 - (int)(input_buffer_x_->tensor_dim_.size()) + axis;
    }

    if ((input_buffer_x_->dim_ == input_buffer_y_->dim_) &&
        (input_buffer_x_->transpose_ == input_buffer_y_->transpose_)) {
        params_fast = 1;
    } else if (ydim[0] == 1 && ydim[1] == 1 && ydim[2] == 1 && ydim[3] == 1) {
        by_num = 1;
    } else if (ydim[0] == 1 && ydim[1] == 1 && ydim[2] == 1 && ydim[3] == xdim[3]) {
        by_channel = 1;
    } else if (ydim[0] == 1 && ydim[1] == 1 && ydim[2] == xdim[1] && ydim[3] == xdim[2]) {
        by_HW = 1;
    } else if (ydim[0] == 1 && ydim[1] == 1 && ydim[2] == 1 && ydim[3] == xdim[2]) {
        by_W = 1;
    } else {
        LOG(FATAL) << ele_type_ << " does not support the current input dimensions.";
    }

    ElementwiseAddMetalParam element_params = {params_fast,
        by_channel,
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
            input_buffer_y_->transpose_[3]},
        by_num,
        by_HW,
        by_W,
        arithmetic_type};

    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(element_params), &element_params);

    function_name_ = fuse_flag_ ? "elementwise_relu" : "elementwise";
    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

#pragma mark - MPS

template <typename T>
void ElementwiseImageCompute::run_with_mps() {
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    auto cmdbuf = [backend commandBuffer];

    if (mps_op_) {
        if (@available(iOS 11.3, macOS 10.13.4, macCatalyst 13.0, *)) {
            ((__bridge T*)mps_op_).primaryStrideInPixelsY = input_buffer_x_->dim_[1] == 1 ? 0 : 1;
            ((__bridge T*)mps_op_).primaryStrideInPixelsX = input_buffer_x_->dim_[2] == 1 ? 0 : 1;
            ((__bridge T*)mps_op_).secondaryStrideInPixelsY = input_buffer_y_->dim_[1] == 1 ? 0 : 1;
            ((__bridge T*)mps_op_).secondaryStrideInPixelsX = input_buffer_y_->dim_[2] == 1 ? 0 : 1;
            [((__bridge T*)mps_op_) encodeToCommandBuffer:cmdbuf
                                             primaryImage:(__bridge MPSImage*)mps_input_image_
                                           secondaryImage:(__bridge MPSImage*)mps_input_image_y_
                                         destinationImage:(__bridge MPSImage*)mps_output_image_];
        }
    }
    [backend commit:cmdbuf];
}

template <typename T>
void ElementwiseImageCompute::setup_with_mps() {
    if (@available(iOS 11.3, macOS 10.13.4, macCatalyst 13.0, *)) {
        auto backend = (__bridge MetalContextImp*)metal_context_->backend();
        mps_op_ = (__bridge_retained void*)[[T alloc] initWithDevice:backend.device];
        // MPS input and output
        auto input_x_c = 4;
        auto input_y_c = 4;
        auto output_c = 4;

        input_x_c = MAX(4, static_cast<int>(input_buffer_x_->dim_[3]));
        input_y_c = MAX(4, static_cast<int>(input_buffer_y_->dim_[3]));
        output_c = MAX(4, static_cast<int>(output_buffer_->dim_[3]));

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
}

void ElementwiseImageCompute::release_memory() {
    TargetWrapperMetal::FreeImage(output_buffer_);
}

void ElementwiseImageCompute::release_mps_memory() {
    if (mps_op_) {
        CFRelease(mps_op_);
        mps_op_ = nullptr;
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
}

ElementwiseImageCompute::~ElementwiseImageCompute() {
    release_memory();
    release_mps_memory();
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
    paddle::lite::kernels::metal::ElementwiseImageCompute,
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
    paddle::lite::kernels::metal::ElementwiseImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(fusion_elementwise_add_activation,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseImageCompute,
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

REGISTER_LITE_KERNEL(fusion_elementwise_add_activation,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseImageCompute,
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

REGISTER_LITE_KERNEL(elementwise_div,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseImageCompute,
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

REGISTER_LITE_KERNEL(elementwise_mul,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseImageCompute,
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

REGISTER_LITE_KERNEL(elementwise_sub,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::ElementwiseImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
