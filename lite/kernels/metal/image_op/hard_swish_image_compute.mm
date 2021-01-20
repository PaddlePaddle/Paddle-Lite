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


#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/hard_swish_image_compute.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void hard_swish_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();

  input_buffer_ = param.X->data<float, metal_image>();
  output_buffer_ = param.Out->mutable_data<float, metal_image>(output_dims);

  HardSwishMetalParam metal_param{
      param.hard_swish_offset, param.hard_swish_threshold, param.hard_swish_scale};
  param_buffer_ = mtl_ctx->create_buffer(
      *device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);

  string function_name = "hard_swish_float";
  kernel_ = mtl_ctx->get_kernel(*device, function_name);
}


void hard_swish_image_compute::Run() {
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    auto args = {metal_kernel_arg{input_buffer_},
                 metal_kernel_arg{output_buffer_},
                 metal_kernel_arg{param_buffer_}};
    kernel_->execute(*queue, global_work_size, false, args);
    queue->wait_until_complete();
  }
}

void hard_swish_image_compute_half::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();

  input_buffer_ = param.X->data<metal_half, metal_image>();
  output_buffer_ = param.Out->mutable_data<metal_half, metal_image>(output_dims);
  HardSwishMetalParam metal_param{
      param.hard_swish_offset, param.hard_swish_threshold, param.hard_swish_scale};
  param_buffer_ = mtl_ctx->create_buffer(
      *device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);

  string function_name = "hard_swish_half";
  kernel_ = mtl_ctx->get_kernel(*device, function_name);
}

void hard_swish_image_compute_half::Run() {
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    auto args = {metal_kernel_arg{input_buffer_},
                 metal_kernel_arg{output_buffer_},
                 metal_kernel_arg{param_buffer_}};

    kernel_->execute(*queue, global_work_size, false, args);
    queue->wait_until_complete();
  }
#if 0
  const auto& param = this->Param<param_t>();
  metal_debug::dump_image("input_buffer_", input_buffer_, param.x->dims().production());
  metal_debug::dump_image("output_buffer_", output_buffer_, param.output->dims().production());
#endif
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(hard_swish,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::hard_swish_image_compute,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();


REGISTER_LITE_KERNEL(hard_swish,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::hard_swish_image_compute_half,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();
