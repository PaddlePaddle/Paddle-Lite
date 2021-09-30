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

#include "lite/kernels/metal/image_op/elementwise_max_image_compute.h"
#include "lite/backends/metal/metal_debug.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename P, PrecisionType PTYPE>
void ElementwiseMaxImageCompute<P, PTYPE>::PrepareForRun() {
    auto& context = this->ctx_->template As<ContextMetal>();
    metal_context_ = (MetalContext*)context.context();
    auto device = metal_context_->GetDefaultDevice();

    const auto& param = this->template Param<param_t>();
    auto output_dims = param.Out->dims();
    auto input_dims = param.X->dims();

    output_buffer_ = param.Out->template mutable_data<P, MetalImage>(output_dims);
    input_buffer_x_ = param.X->template data<P, MetalImage>();
    input_buffer_y_ = param.Y->template data<P, MetalImage>();

    bool valid = false;
    int by_channel = 0;
    if (input_buffer_x_->tensor_dim_.size() == 4 && input_buffer_y_->tensor_dim_.size() == 4 &&
        param.axis == -1 && input_buffer_y_->tensor_dim_[2] == 1 &&
        input_buffer_y_->tensor_dim_[3] == 1) {
        by_channel = 1;
        valid = true;
    } else if (input_buffer_x_->tensor_dim_.size() == input_buffer_y_->tensor_dim_.size()) {
        valid = true;
        for (int i = 0; i < input_buffer_x_->tensor_dim_.size(); i++) {
            if (input_buffer_x_->tensor_dim_[i] != input_buffer_y_->tensor_dim_[i]) {
                valid = false;
                break;
            }
        }
        if (valid) {
            by_channel = 0;
        }
    }
    if (!valid) {
        throw std::logic_error(
            "ERROR: elementwise_sub only supports : 1. input shapes are the same. "
            "2. multiply by channel.");
    }

    ElementwiseMetalParam element_params = {by_channel};

    params_buffer_ = metal_context_->CreateBuffer(
        *device, &element_params, sizeof(element_params), METAL_ACCESS_FLAG::CPUWriteOnly);
    std::string function_name = "";
    if (std::is_same<float, P>::value) {
        function_name = "elementwise_max";
    } else if (std::is_same<MetalHalf, P>::value) {
        function_name = "elementwise_max_half";
    }

    queue_ = metal_context_->GetDefaultQueue(*device);
    kernel_ = metal_context_->GetKernel(*device, function_name);
}

template <typename P, PrecisionType PTYPE>
void ElementwiseMaxImageCompute<P, PTYPE>::Run() {
    const auto& param = this->template Param<param_t>();
    auto output_width = output_buffer_->texture_width_;
    auto output_height = output_buffer_->texture_height_;
    auto output_array_length = output_buffer_->array_length_;

    auto encoder =
        std::make_shared<MetalEncoder>(metal_context_->cmd_buf_.get(), &kernel_->program_);
    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
        static_cast<MetalUint>(output_height),
        static_cast<MetalUint>(output_array_length)};

    [encoder->metal_command_encoder_ setTexture:(input_buffer_x_->image()) atIndex:(0)];
    [encoder->metal_command_encoder_ setTexture:(input_buffer_y_->image()) atIndex:(1)];
    [encoder->metal_command_encoder_ setTexture:(output_buffer_->image()) atIndex:(2)];
    [encoder->metal_command_encoder_ setBuffer:(params_buffer_->buffer()) offset:(0) atIndex:(0)];

    kernel_->Execute(*encoder, global_work_size, false);
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

template class paddle::lite::kernels::metal::ElementwiseMaxImageCompute<float, PRECISION(kFloat)>;
template class paddle::lite::kernels::metal::ElementwiseMaxImageCompute<MetalHalf,
    PRECISION(kFP16)>;

typedef paddle::lite::kernels::metal::ElementwiseMaxImageCompute<float, PRECISION(kFloat)>
    MetalElementwiseMaxFp32;
typedef paddle::lite::kernels::metal::ElementwiseMaxImageCompute<MetalHalf, PRECISION(kFP16)>
    MetalElementwiseMaxFp16;

REGISTER_LITE_KERNEL(elementwise_max,
    kMetal,
    kFloat,
    kMetalTexture2DArray,
    MetalElementwiseMaxFp32,
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

REGISTER_LITE_KERNEL(elementwise_max,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    MetalElementwiseMaxFp16,
    def)
    .BindInput("X",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Y",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
        {LiteType::GetTensorTy(TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();