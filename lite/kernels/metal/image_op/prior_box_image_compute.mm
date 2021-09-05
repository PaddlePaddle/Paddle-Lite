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

#include "lite/kernels/metal/image_op/prior_box_image_compute.h"
#include "lite/backends/metal/metal_context_imp.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void PriorBoxImageCompute::PrepareForRun() {
    auto& context = ctx_->As<MTLContext>();
    metal_context_ = (MetalContext*)context.context();

    const auto& param = this->Param<param_t>();
    auto box_dims = param.boxes->dims();
#ifdef LITE_WITH_METAL_FULL
#else
    input_buffer_ = param.input->data<MetalHalf, MetalImage>();
    image_buffer_ = param.image->data<MetalHalf, MetalImage>();
    output_box_ = param.boxes->mutable_data<MetalHalf, MetalImage>(metal_context_, box_dims);
    output_variances_ = param.boxes->mutable_data<MetalHalf, MetalImage>(metal_context_, box_dims);

#endif

    assert(param.min_sizes.size() == 1);
    auto image_width = static_cast<float>(image_buffer_->pad_to_four_dim_[3]);
    auto image_height = static_cast<float>(image_buffer_->pad_to_four_dim_[2]);
    auto feature_width = static_cast<float>(input_buffer_->pad_to_four_dim_[3]);
    auto feature_height = static_cast<float>(input_buffer_->pad_to_four_dim_[2]);

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

    new_aspect_ratio_buffer_ = std::make_shared<MetalBuffer>(
        metal_context_, output_aspect_ratios.size() * sizeof(float), static_cast<void*>(output_aspect_ratios.data()));
    variances_buffer_ = std::make_shared<MetalBuffer>(
       metal_context_, param.variances_.size() * sizeof(float), static_cast<void*>(param.variances_.data()));

    auto max_sizes_size = (uint32_t)(param.max_sizes.size());
    auto min_sizes_size = (uint32_t)(param.min_sizes.size());

    auto num_priors = aspect_ratios_size * min_sizes_size + max_sizes_size;

    float minSize = (bool)(*(param.min_sizes.end())) ? *(param.min_sizes.end()) : 0.0f;
    float maxSize = (bool)(*(param.max_sizes.end())) ? *(param.min_sizes.end()) : 0.0f;

    PriorBoxMetalParam metal_param = {param.offset,
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
    param_buffer_ =
        std::make_shared<MetalBuffer>(metal_context_, sizeof(metal_param), &metal_param);

    function_name_ = "prior_box";

    if (param.min_max_aspect_ratios_order) function_name_ = "prior_box_MinMaxAspectRatiosOrder";

    // pipline
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();
    pipline_ = [backend pipline:function_name_];
}

void PriorBoxImageCompute::Run() {
    @autoreleasepool {
        run_without_mps();
    }
}

void PriorBoxImageCompute::run_without_mps() {
    auto pipline = pipline_;
    auto outTexture = output_box_->image();
    auto backend = (__bridge MetalContextImp*)metal_context_->backend();

    auto encoder = [backend commandEncoder];
    [encoder setTexture:(input_buffer_->image()) atIndex:(0)];
    [encoder setTexture:(output_box_->image()) atIndex:(1)];
    [encoder setTexture:(output_variances_->image()) atIndex:(2)];

    [encoder setBuffer:(new_aspect_ratio_buffer_->buffer()) offset:(0) atIndex:(1)];
    [encoder setBuffer:(param_buffer_->buffer()) offset:(0) atIndex:(1)];
    [encoder setBuffer:(variances_buffer_->buffer()) offset:(0) atIndex:(2)];

    [backend dispatchEncoder:encoder pipline:pipline outTexture:outTexture];
    [backend commit];
}

PriorBoxImageCompute::~PriorBoxImageCompute() {
    TargetWrapperMetal::FreeImage(output_box_);
    TargetWrapperMetal::FreeImage(output_variances_);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(prior_box,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::PriorBoxImageCompute,
    def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Image",
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

REGISTER_LITE_KERNEL(prior_box,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::PriorBoxImageCompute,
    def)
    .BindInput("Input",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Image",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Boxes",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Variances",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
