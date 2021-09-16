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

#include "lite/kernels/metal/image_op/prior_box_image_compute.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void PriorBoxImageCompute<P, PTYPE>::PrepareForRun() {
    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();
    auto device = metal_context_->GetDefaultDevice();

    const auto& param = this->template Param<param_t>();
    auto box_dims = param.boxes->dims();
    auto variances_dims = param.variances->dims();

    input_buffer_ = param.input->template data<P, MetalImage>();
    image_buffer_ = param.image->template data<P, MetalImage>();
    output_buffer_ = param.boxes->template mutable_data<P, MetalImage>(box_dims);
    variances_buffer_ = param.variances->template mutable_data<P, MetalImage>(variances_dims);

    assert(param.min_sizes.size() == 1);
    auto image_width = (float)(image_buffer_->pad_to_four_dim_[3]);
    auto image_height = (float)(image_buffer_->pad_to_four_dim_[2]);
    auto feature_width = (float)(input_buffer_->pad_to_four_dim_[3]);
    auto feature_height = (float)(input_buffer_->pad_to_four_dim_[2]);

    float step_w = param.step_w;
    float step_h = param.step_h;
    if (step_w == 0 || step_h == 0) {
        step_w = image_width / feature_width;
        step_h = image_height / feature_height;
    }

    std::vector<float> output_aspect_ratios{};
    output_aspect_ratios.push_back(0.1);
    auto epsilon = 1e-6;
    for (auto ar : param.aspect_ratios) {
        auto already_exist = false;
        for (auto outputAr : output_aspect_ratios) {
            if (fabs(double(ar) - double(outputAr)) < epsilon) {
                already_exist = true;
                break;
            }
        }

        if (!already_exist) {
            output_aspect_ratios.push_back(ar);
        }
        if (param.flip) {
            output_aspect_ratios.push_back(1.0f / ar);
        }
    }

    auto aspect_ratios_size = (uint32_t)(output_aspect_ratios.size());

    new_aspect_ratio_buffer_ = metal_context_->CreateBuffer(*device,
        output_aspect_ratios.data(),
        aspect_ratios_size * sizeof(float),
        METAL_ACCESS_FLAG::CPUWriteOnly);

    auto max_sizes_size = (uint32_t)(param.max_sizes.size());
    auto min_sizes_size = (uint32_t)(param.min_sizes.size());

    auto num_priors = aspect_ratios_size * min_sizes_size + max_sizes_size;

    float minSize = (bool)(*(param.min_sizes.end())) ? *(param.min_sizes.end()) : 0.0f;
    float maxSize = (bool)(*(param.max_sizes.end())) ? *(param.min_sizes.end()) : 0.0f;

    PriorBoxMetalParam prior_box_param = {param.offset,
        step_w,
        step_h,
        minSize,
        maxSize,
        image_width,
        image_height,
        param.clip,
        num_priors,
        aspect_ratios_size,
        min_sizes_size,
        max_sizes_size};

    new_aspect_ratio_buffer_ = metal_context_->CreateBuffer(
        *device, &prior_box_param, sizeof(prior_box_param), METAL_ACCESS_FLAG::CPUWriteOnly);

    std::string function_name = "";
    if (std::is_same<float, P>::value) {
        function_name = "prior_box";
        if (param.min_max_aspect_ratios_order) function_name = "prior_box_MinMaxAspectRatiosOrder";
    } else if (std::is_same<MetalHalf, P>::value) {
        function_name = "prior_box_half";
        if (param.min_max_aspect_ratios_order)
            function_name = "prior_box_MinMaxAspectRatiosOrder_half";
    }
    assert(!function_name.empty());

    kernel_ = metal_context_->GetKernel(*device, function_name);
    queue_ = metal_context_->GetDefaultQueue(*device);
}

template <typename P, PrecisionType PTYPE>
void PriorBoxImageCompute<P, PTYPE>::Run() {
    const auto& param = this->template Param<param_t>();
    auto output_width = output_buffer_->texture_width_;
    auto output_height = output_buffer_->texture_height_;
    auto output_array_length = output_buffer_->array_length_;

    auto encoder =
        std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
        static_cast<MetalUint>(output_height),
        static_cast<MetalUint>(output_array_length)};

    [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
    [encoder->metal_command_encoder_ setTexture:(image_buffer_->image()) atIndex:(1)];
    [encoder->metal_command_encoder_ setBuffer:(new_aspect_ratio_buffer_->buffer())
                                        offset:(0)
                                       atIndex:(0)];
    [encoder->metal_command_encoder_ setBuffer:(param_buffer_->buffer()) offset:(0) atIndex:(1)];
    [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(2)];
    [encoder->metal_command_encoder_ setBytes:(param.variances_.data())
                                       length:(sizeof(float) * param.variances_.size())
                                      atIndex:(2)];
    kernel_->Execute(*encoder, global_work_size, false);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::PriorBoxImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::PriorBoxImageCompute<MetalHalf, PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::PriorBoxImageCompute<float, PRECISION(kFloat)>
    MetalPriorBoxFp32;
typedef paddle::lite::kernels::metal::PriorBoxImageCompute<MetalHalf, PRECISION(kFP16)>
    MetalPriorBoxFp16;

REGISTER_LITE_KERNEL(prior_box, kMetal, kFloat, kMetalTexture2DArray, MetalPriorBoxFp32, def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Image",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Boxes",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Variances",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(prior_box, kMetal, kFP16, kMetalTexture2DArray, MetalPriorBoxFp16, def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Image",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Boxes",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Variances",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
