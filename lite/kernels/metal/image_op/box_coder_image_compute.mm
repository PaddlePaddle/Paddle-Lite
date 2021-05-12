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

#include "lite/kernels/metal/image_op/box_coder_image_compute.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"
#include <algorithm>

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void BoxCoderImageCompute<P, PTYPE>::PrepareForRun() {
    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();
    auto device = metal_context_->GetDefaultDevice();

    const auto& param = this->template Param<param_t>();
    auto output_dims = param.proposals->dims();
    auto prior_box_dims = param.prior_box->dims();

    assert(param.code_type == "decode_center_size" && param.box_normalized == true);

    prior_box_buffer_ = param.prior_box->template data<P, MetalImage>();
    prior_box_var_buffer_ = param.prior_box_var->template data<P, MetalImage>();
    target_box_buffer_ = param.target_box->template data<P, MetalImage>();
    output_buffer_ = param.proposals->template mutable_data<P, MetalImage>(output_dims);

    std::string function_name = "";
    if (std::is_same<float, P>::value) {
        function_name = "boxcoder_float";
    } else if (std::is_same<MetalHalf, P>::value) {
        function_name = "boxcoder_half";
    }
    assert(!function_name.empty());

    queue_ = metal_context_->GetDefaultQueue(*device);
    kernel_ = metal_context_->GetKernel(*device, function_name);
}

template <typename P, PrecisionType PTYPE>
void BoxCoderImageCompute<P, PTYPE>::Run() {
    auto output_width = output_buffer_->texture_width_;
    auto output_height = output_buffer_->texture_height_;
    auto output_array_length = output_buffer_->array_length_;

    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();

    auto encoder =
        std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                   static_cast<MetalUint>(output_height),
                                   static_cast<MetalUint>(output_array_length)};

    [encoder->metal_command_encoder_ setTexture:(prior_box_buffer_->image()) atIndex:(0)];
    [encoder->metal_command_encoder_ setTexture:(prior_box_var_buffer_->image()) atIndex:(1)];
    [encoder->metal_command_encoder_ setTexture:(target_box_buffer_->image()) atIndex:(2)];
    [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(3)];

    kernel_->Execute(*encoder, global_work_size, false);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::BoxCoderImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::BoxCoderImageCompute<MetalHalf, PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::BoxCoderImageCompute<float, PRECISION(kFloat)>
    MetalBoxCoderFp32;
typedef paddle::lite::kernels::metal::BoxCoderImageCompute<MetalHalf, PRECISION(kFP16)>
    MetalBoxCoderFp16;

REGISTER_LITE_KERNEL(box_coder, kMetal, kFloat, kMetalTexture2DArray, MetalBoxCoderFp32, def)
    .BindInput("PriorBox",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("PriorBoxVar",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("TargetBox",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("OutputBox",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(box_coder, kMetal, kFP16, kMetalTexture2DArray, MetalBoxCoderFp16, def)
    .BindInput(
        "PriorBox",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput(
        "PriorBoxVar",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput(
        "TargetBox",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput(
        "OutputBox",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
