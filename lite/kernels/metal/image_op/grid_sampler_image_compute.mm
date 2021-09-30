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

#include "lite/kernels/metal/image_op/grid_sampler_image_compute.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void GridSamplerImageCompute<P, PTYPE>::PrepareForRun() {
    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();
    auto device = metal_context_->GetDefaultDevice();

    const auto& param = this->template Param<param_t>();
    auto output_dims = param.out->dims();

    input_buffer_ = param.x->template data<P, MetalImage>();
    grid_buffer_ = param.x->template data<float, MetalImage>();
    output_buffer_ = param.out->template mutable_data<P, MetalImage>(output_dims);
    std::vector<int> excepted_transpose = {1, 2, 3, 4};
    if (grid_buffer_->transpose_ != excepted_transpose) {
        insert_shape = true;
        std::unique_ptr<KernelContext> reshape_ctx(new KernelContext);
        reshape_ctx->template As<ContextMetal>().InitOnce();
        operators::ReshapeParam reshape_param;
        reshape_param.x = param.x;
        reshape_param.excepted_transpose_ = excepted_transpose;
        shape_out_dev.Resize(grid_buffer_->tensor_dim_);
        reshape_param.output = &shape_out_dev;
        reshape_.SetContext(std::move(reshape_ctx));
        reshape_.SetParam(reshape_param);
        reshape_.PrepareForRun();
    }

    std::string function_name = "";
    if (std::is_same<float, P>::value) {
        function_name = "grid_sampler";
    } else if (std::is_same<MetalHalf, P>::value) {
        function_name = "grid_sampler_half";
    }
    assert(!function_name.empty());

    queue_ = metal_context_->GetDefaultQueue(*device);
    kernel_ = metal_context_->GetKernel(*device, function_name);
}

template <typename P, PrecisionType PTYPE>
void GridSamplerImageCompute<P, PTYPE>::Run() {
    auto output_width = output_buffer_->texture_width_;
    auto output_height = output_buffer_->texture_height_;
    auto output_array_length = output_buffer_->array_length_;

    auto encoder =
        std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
        static_cast<MetalUint>(output_height),
        static_cast<MetalUint>(output_array_length)};

    if (insert_shape) {
        reshape_.Run();
        auto shape_buffer = shape_out_dev.data<float, MetalImage>();
        [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
        [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(1)];
        [encoder->metal_command_encoder_ setTexture:(shape_buffer->image()) atIndex:(2)];
    } else {
        [encoder->metal_command_encoder_ setTexture:(input_buffer_->image()) atIndex:(0)];
        [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(1)];
        [encoder->metal_command_encoder_ setTexture:(grid_buffer_->image()) atIndex:(2)];
    }

    kernel_->Execute(*encoder, global_work_size, false);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::GridSamplerImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::GridSamplerImageCompute<MetalHalf, PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::GridSamplerImageCompute<float, PRECISION(kFloat)>
    MetalGridSamplerFp32;
typedef paddle::lite::kernels::metal::GridSamplerImageCompute<MetalHalf, PRECISION(kFP16)>
    MetalGridSamplerFp16;

REGISTER_LITE_KERNEL(grid_sampler, kMetal, kFloat, kMetalTexture2DArray, MetalGridSamplerFp32, def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Grid",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(grid_sampler, kMetal, kFP16, kMetalTexture2DArray, MetalGridSamplerFp16, def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Grid",
        {LiteType::GetTensorTy(TARGET(kMetal),
            PRECISION(kFloat),
            DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
