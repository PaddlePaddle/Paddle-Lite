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

#include "lite/kernels/metal/image_op/interp_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void BilinearInterpImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.Out->dims();
#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.X->data<MetalHalf, MetalImage>();
    output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif

    setup_without_mps();
}

void BilinearInterpImageCompute::Run() {
    @autoreleasepool {
        run_without_mps();
    }
}

void BilinearInterpImageCompute::run_without_mps() {
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

void BilinearInterpImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();

    int input_h = static_cast<int>(input_buffer_->pad_to_four_dim_[2]);
    int input_w = static_cast<int>(input_buffer_->pad_to_four_dim_[3]);
    int output_h = static_cast<int>(output_buffer_->pad_to_four_dim_[2]);
    int output_w = static_cast<int>(output_buffer_->pad_to_four_dim_[3]);

    float delta_h = 0;
    float delta_w = 0;

    if (param.align_corners && output_h > 1) {
        delta_h = 1.0;
    }
    if (param.align_corners && output_w > 1) {
        delta_w = 1.0;
    }
    float ratio_h = ((float)(input_h)-delta_h) / ((float)(output_h)-delta_h);
    float ratio_w = ((float)(input_w)-delta_w) / ((float)(output_w)-delta_w);

    float align_delta = 0;
    bool align_flag = (param.align_mode == 0 && !param.align_corners);

    if (align_flag) {
        align_delta = 0.5;
    }

    BilinearInterPMetalParam metal_param{ratio_h, ratio_w, align_delta};
    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(metal_param), &metal_param);

    function_name_ = "bilinear_interp";
    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

BilinearInterpImageCompute::~BilinearInterpImageCompute() {
    TargetWrapperMetal::FreeImage(output_buffer_);
}

void NearestInterpImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.Out->dims();
#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.X->data<MetalHalf, MetalImage>();
    output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif

    setup_without_mps();
}

void NearestInterpImageCompute::Run() {
    @autoreleasepool {
        run_without_mps();
    }
}

void NearestInterpImageCompute::run_without_mps() {
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

void NearestInterpImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();

    int input_h = static_cast<int>(input_buffer_->pad_to_four_dim_[2]);
    int input_w = static_cast<int>(input_buffer_->pad_to_four_dim_[3]);
    int output_h = static_cast<int>(output_buffer_->pad_to_four_dim_[2]);
    int output_w = static_cast<int>(output_buffer_->pad_to_four_dim_[3]);

    float ratio_w = 1.0f;
    float ratio_h = 1.0f;
    float align_delta = 0.0f;
    if (param.align_corners) {
        ratio_w = (float(input_w) - 1.0f) / (float(output_w) - 1.0f);
        ratio_h = (float(input_h) - 1.0f) / (float(output_h) - 1.0f);
        align_delta = 0.5f;
    } else {
        ratio_w = float(input_w) / float(output_w);
        ratio_h = float(input_h) / float(output_h);
        align_delta = 0.0;
    }

    NearestInterpMetalParam interp_param{ratio_h, ratio_w, align_delta};
    params_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(interp_param), &interp_param);

    function_name_ = "nearest_interp";
    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

NearestInterpImageCompute::~NearestInterpImageCompute() {
    TargetWrapperMetal::FreeImage(output_buffer_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(bilinear_interp,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::BilinearInterpImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("OutSize",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(bilinear_interp,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::BilinearInterpImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("OutSize",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(bilinear_interp_v2,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::BilinearInterpImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("OutSize",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(bilinear_interp_v2,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::BilinearInterpImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("OutSize",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kInt32), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::NearestInterpImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("OutSize",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::NearestInterpImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("OutSize",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp_v2,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::NearestInterpImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("OutSize",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp_v2,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::NearestInterpImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("OutSize",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
