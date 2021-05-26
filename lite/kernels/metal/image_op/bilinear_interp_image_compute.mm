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

#include "bilinear_interp_image_compute.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void BilinearInterpImageCompute<P, PTYPE>::PrepareForRun() {
    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();
    auto device = metal_context_->GetDefaultDevice();

    const auto& param = this->template Param<param_t>();
    auto output_dims = param.Out->dims();

    input_buffer_ = param.X->template data<P, MetalImage>();
    output_buffer_ =
        param.Out->template mutable_data<P, MetalImage>(output_dims, input_buffer_->transpose_);

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

    param_buffer_ = metal_context_->CreateBuffer(*device, &metal_param, sizeof(metal_param),
                                                 METAL_ACCESS_FLAG::CPUWriteOnly);

    std::string function_name = "";
    if (std::is_same<float, P>::value) {
        function_name = "bilinear_interp_float";
    } else if (std::is_same<MetalHalf, P>::value) {
        function_name = "bilinear_interp_half";
    }

    queue_ = metal_context_->GetDefaultQueue(*device);
    kernel_ = metal_context_->GetKernel(*device, function_name);
}

template <typename P, PrecisionType PTYPE>
void BilinearInterpImageCompute<P, PTYPE>::Run() {
    auto output_width = output_buffer_->texture_width_;
    auto output_height = output_buffer_->texture_height_;
    auto output_array_length = output_buffer_->array_length_;

    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();

    {
        auto encoder =
            std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
        MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                       static_cast<MetalUint>(output_height),
                                       static_cast<MetalUint>(output_array_length)};

        [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
        [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(1)];
        [encoder->metal_command_encoder_ setBuffer:(param_buffer_->buffer()) offset:(0)atIndex:(0)];

        kernel_->Execute(*encoder, global_work_size, false);
    }
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::BilinearInterpImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::BilinearInterpImageCompute<MetalHalf,
                                                                        PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::BilinearInterpImageCompute<float, PRECISION(kFloat)>
    MetalBilinearInterpFp32;
typedef paddle::lite::kernels::metal::BilinearInterpImageCompute<MetalHalf, PRECISION(kFP16)>
    MetalBilinearInterpFp16;

REGISTER_LITE_KERNEL(
    bilinear_interp, kMetal, kFloat, kMetalTexture2DArray, MetalBilinearInterpFp32, def)
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

REGISTER_LITE_KERNEL(
    bilinear_interp, kMetal, kFP16, kMetalTexture2DArray, MetalBilinearInterpFp16, def)
    .BindInput(
        "X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindInput("Scale",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW))})
    .BindOutput(
        "Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();