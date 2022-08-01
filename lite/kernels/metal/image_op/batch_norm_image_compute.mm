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

#include "lite/kernels/metal/image_op/batch_norm_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void BatchNormImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto output_dims = param.y->dims();

#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.x->data<MetalHalf, MetalImage>();
    output_buffer_ = param.y->mutable_data<MetalHalf, MetalImage>(metal_context_, output_dims);
#endif

    setup_without_mps();
}

void BatchNormImageCompute::Run() {
    @autoreleasepool {
        run_without_mps();
    }
}

void BatchNormImageCompute::run_without_mps() {
    auto pipline = pipline_;
    auto outTexture = output_buffer_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:input_buffer_->image() atIndex:(0)];
    [encoder setTexture:(output_buffer_->image()) atIndex:(1)];
    [encoder setBuffer:(scale_buffer_->buffer()) offset:(0) atIndex:(0)];
    [encoder setBuffer:(bias_buffer_->buffer()) offset:(0) atIndex:(1)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

void BatchNormImageCompute::setup_without_mps() {
    const auto& param = this->Param<param_t>();
    auto input_num = input_buffer_->tensor_dim_[0];
    auto bias_dims = param.bias->dims();
    auto scale_dims = param.scale->dims();
    auto count = param.variance->dims().production();
    bias_dims[0] = bias_dims[0] * input_num;
    scale_dims[0] = scale_dims[0] * input_num;
    CHECK_EQ(bias_dims.production(), count * input_num) << "batchnorm: param error";
    CHECK_EQ(scale_dims.production(), count * input_num) << "batchnorm: param error";

    auto mean_raw_ptr = param.mean->template data<float>();
    auto bias_raw_ptr = param.bias->template data<float>();
    auto scale_raw_ptr = param.scale->template data<float>();
    auto variance_raw_ptr = param.variance->template data<float>();

    bias_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, bias_dims, METAL_PRECISION_TYPE::HALF);
    scale_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, scale_dims, METAL_PRECISION_TYPE::HALF);

    auto scale_ptr = (float*)TargetWrapperHost::Malloc(sizeof(float) * input_num * count);
    auto bias_ptr = (float*)TargetWrapperHost::Malloc(sizeof(float) * input_num * count);
    for (int i = 0; i < input_num * count; i++) {
        int j = i % count;
        auto inv_std = 1.0f / std::sqrt(variance_raw_ptr[j] + param.epsilon);
        bias_ptr[i] = bias_raw_ptr[j] - mean_raw_ptr[j] * inv_std * scale_raw_ptr[j];
        scale_ptr[i] = inv_std * scale_raw_ptr[j];
    }
    bias_buffer_->CopyFromNCHW<float>(bias_ptr);
    scale_buffer_->CopyFromNCHW<float>(scale_ptr);

    TargetWrapperHost::Free(bias_ptr);
    TargetWrapperHost::Free(scale_ptr);

    function_name_ = "batchnorm";
    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

BatchNormImageCompute::~BatchNormImageCompute() {
    TargetWrapperMetal::FreeImage(output_buffer_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(batch_norm,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::BatchNormImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Variance",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("VarianceOut",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("SavedMean",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("SavedVariance",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("MeanOut",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(batch_norm,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::BatchNormImageCompute,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Variance",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("VarianceOut",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("SavedMean",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("SavedVariance",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("MeanOut",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
