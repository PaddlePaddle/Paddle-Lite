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

#include "lite/kernels/metal/image_op/pixel_shuffle_image_compute.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void PixelShuffleImageCompute<P, PTYPE>::PrepareForRun() {
    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();
    auto device = metal_context_->GetDefaultDevice();

    const auto& param = this->template Param<param_t>();
    auto output_dims = param.output->dims();

    input_buffer_ = param.x->template data<P, MetalImage>();
    output_buffer_ = param.output->template mutable_data<P, MetalImage>(output_dims);

    PixelShuffleMetalParam metal_param{param.upscale_factor};
    param_buffer_ = metal_context_->CreateBuffer(
        *device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);

    std::string function_name = "";
    if (std::is_same<float, P>::value) {
        function_name = "pixel_shuffle";
    } else if (std::is_same<MetalHalf, P>::value) {
        function_name = "pixel_shuffle_half";
    }
    assert(!function_name.empty());

    kernel_ = metal_context_->GetKernel(*device, function_name);
    queue_ = metal_context_->GetDefaultQueue(*device);
}

template <typename P, PrecisionType PTYPE>
void PixelShuffleImageCompute<P, PTYPE>::Run() {
    auto output_width = output_buffer_->texture_width_;
    auto output_height = output_buffer_->texture_height_;
    auto output_array_length = output_buffer_->array_length_;

    auto encoder =
        std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
        static_cast<MetalUint>(output_height),
        static_cast<MetalUint>(output_array_length)};

    [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
    [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(1)];
    [encoder->metal_command_encoder_ setBuffer:(param_buffer_->buffer()) offset:(0) atIndex:(0)];

    kernel_->Execute(*encoder, global_work_size, false);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::PixelShuffleImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::PixelShuffleImageCompute<MetalHalf, PRECISION(kFP16)>;
typedef paddle::lite::kernels::metal::PixelShuffleImageCompute<float, PRECISION(kFloat)>
    MetalPixelShuffleFp32;
typedef paddle::lite::kernels::metal::PixelShuffleImageCompute<MetalHalf, PRECISION(kFP16)>
    MetalPixelShuffleFp16;

REGISTER_LITE_KERNEL(pixel_shuffle,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    MetalPixelShuffleFp32,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(pixel_shuffle, kMetal, kFP16, kMetalTexture2DArray, MetalPixelShuffleFp16, def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
