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

#include "lite/kernels/metal/image_op/instance_norm_image_compute.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void InstanceNormImageCompute<P, PTYPE>::PrepareForRun() {
    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();
    auto device = metal_context_->GetDefaultDevice();

    const auto& param = this->template Param<param_t>();
    auto output_dims = param.x->dims();
    auto input_dims = param.x->dims();
    auto scale_dims = param.scale->dims();
    auto bias_dims = param.bias->dims();

    output_buffer_ = param.out->template mutable_data<P, MetalImage>(param.out->dims());
    input_buffer_ = param.x->template data<P, MetalImage>();

    uint16_t has_relu = (uint16_t)param.fuse_relu;
    InstanceNormReluMetalParam metal_param{has_relu};
    params_buffer_ = metal_context_->CreateBuffer(
        *device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);

    auto bias_raw_buffer = param.bias->template data<float>();
    auto scale_raw_buffer = param.scale->template data<float>();
    auto mean_raw_buffer = param.saved_mean->template data<float>();
    auto variance_ptr = param.saved_variance->template data<float>();

    auto count = scale_dims.production();

    if (std::is_same<P, float>::value) {
        scale_buffer_ =
            std::make_shared<MetalBuffer>(*device, scale_dims, METAL_PRECISION_TYPE::FLOAT, true);
        bias_buffer_ =
            std::make_shared<MetalBuffer>(*device, bias_dims, METAL_PRECISION_TYPE::FLOAT, true);

        float* scale_buffer = (float*)malloc(count * sizeof(float));
        float* bias_buffer = (float*)malloc(count * sizeof(float));

        for (int i = 0; i < count; i++) {
            auto inv_std = 1.0f / std::sqrt(variance_ptr[i] + param.epsilon);
            bias_buffer[i] =
                bias_raw_buffer[i] - mean_raw_buffer[i] * inv_std * scale_raw_buffer[i];
            scale_buffer[i] = inv_std * scale_raw_buffer[i];
        }

        scale_buffer_->CopyFromNCHW<float>(scale_buffer);
        bias_buffer_->CopyFromNCHW<float>(bias_buffer);

        free(scale_buffer);
        free(bias_buffer);
    } else if (std::is_same<MetalHalf, P>::value) {
        scale_buffer_ =
            std::make_shared<MetalBuffer>(*device, scale_dims, METAL_PRECISION_TYPE::HALF, true);
        bias_buffer_ =
            std::make_shared<MetalBuffer>(*device, bias_dims, METAL_PRECISION_TYPE::HALF, true);

        MetalHalf* scale_buffer = (MetalHalf*)malloc(count * sizeof(MetalHalf));
        MetalHalf* bias_buffer = (MetalHalf*)malloc(count * sizeof(MetalHalf));

        for (int i = 0; i < count; i++) {
            auto inv_std = 1.0f / std::sqrt(variance_ptr[i] + param.epsilon);
            bias_buffer[i] = MetalFloat2Half(
                bias_raw_buffer[i] - mean_raw_buffer[i] * inv_std * scale_raw_buffer[i]);
            scale_buffer[i] = MetalFloat2Half(inv_std * scale_raw_buffer[i]);
        }

        scale_buffer_->CopyFromNCHW<MetalHalf>(scale_buffer);
        bias_buffer_->CopyFromNCHW<MetalHalf>(bias_buffer);

        free(scale_buffer);
        free(bias_buffer);
    } else {
        throw std::logic_error("ERROR: instance_norm support fp32 and fp16");
    }

    std::string function_name = "";
    if (std::is_same<float, P>::value) {
        function_name = "instance_norm_relu";
    } else if (std::is_same<MetalHalf, P>::value) {
        function_name = "instance_norm_relu_half";
    }

    assert(!function_name.empty());
    queue_ = metal_context_->GetDefaultQueue(*device);
    kernel_ = metal_context_->GetKernel(*device, function_name);
}

template <typename P, PrecisionType PTYPE>
void InstanceNormImageCompute<P, PTYPE>::Run() {
    const auto& param = this->template Param<param_t>();
    auto input_dims = param.x->dims();
    auto output_dims = param.out->dims();
    auto output_width = output_dims[3];
    auto output_height = output_dims[2];
    auto output_array_length = (output_dims[0] * output_dims[1] + 3) / 4;

    auto encoder =
        std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
        static_cast<MetalUint>(output_height),
        static_cast<MetalUint>(output_array_length)};

    [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
    [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(1)];
    [encoder->metal_command_encoder_ setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];

    kernel_->Execute(*encoder, global_work_size, false);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::InstanceNormImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::InstanceNormImageCompute<MetalHalf, PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::InstanceNormImageCompute<float, PRECISION(kFloat)>
    MetalInstanceNormFp32;
typedef paddle::lite::kernels::metal::InstanceNormImageCompute<MetalHalf, PRECISION(kFP16)>
    MetalInstanceNormFp16;

REGISTER_LITE_KERNEL(instance_norm,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    MetalInstanceNormFp32,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("SavedMean",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("SavedVariance",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(instance_norm, kMetal, kFP16, kMetalTexture2DArray, MetalInstanceNormFp16, def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("SavedMean",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("SavedVariance",
        {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();