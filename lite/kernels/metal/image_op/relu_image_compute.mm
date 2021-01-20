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


#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/relu_image_compute.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void relu_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();
  auto input_dims = param.X->dims();

  output_buffer_ = param.Out->mutable_data<float, metal_image>(output_dims);
  input_buffer_ = param.X->data<float, metal_image>();
}

void relu_image_compute::Run() {
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    string function_name = "relu";
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    kernel_ = mtl_ctx->get_kernel(*mtl_dev, function_name);

    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    auto args = {metal_kernel_arg{input_buffer_}, metal_kernel_arg{output_buffer_}};

    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

void relu6_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();
  auto input_dims = param.X->dims();

  Relu6MetalParam metal_param{param.hard_swish_threshold};
  param_buffer_ = mtl_ctx->create_buffer(
      *device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);

  output_buffer_ = param.Out->mutable_data<float, metal_image>(output_dims);
  input_buffer_ = param.X->data<float, metal_image>();
}

void relu6_image_compute::Run() {
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    string function_name = "relu6";
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    kernel_ = mtl_ctx->get_kernel(*mtl_dev, function_name);

    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    auto args = {metal_kernel_arg{input_buffer_},
                 metal_kernel_arg{output_buffer_},
                 metal_kernel_arg{param_buffer_}};
    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

void relu_image_compute_half::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();
  auto input_dims = param.X->dims();

  output_buffer_ = param.Out->mutable_data<metal_half, metal_image>(output_dims);
  input_buffer_ = param.X->data<metal_half, metal_image>();
}

void relu_image_compute_half::Run() {
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    string function_name = "relu_half";
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    kernel_ = mtl_ctx->get_kernel(*mtl_dev, function_name);

    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    auto args = {metal_kernel_arg{input_buffer_}, metal_kernel_arg{output_buffer_}};

    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

void relu6_image_compute_half::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();
  auto input_dims = param.X->dims();

  Relu6MetalParam metal_param{param.hard_swish_threshold};
  param_buffer_ = mtl_ctx->create_buffer(
      *device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);

  output_buffer_ = param.Out->mutable_data<metal_half, metal_image>(output_dims);
  input_buffer_ = param.X->data<metal_half, metal_image>();
}

void relu6_image_compute_half::Run() {
  auto output_width = output_buffer_->textureWidth_;
  auto output_height = output_buffer_->textureHeight_;
  auto output_array_length = output_buffer_->arrayLength_;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    string function_name = "relu6_half";
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    kernel_ = mtl_ctx->get_kernel(*mtl_dev, function_name);

    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    auto args = {metal_kernel_arg{input_buffer_},
                 metal_kernel_arg{output_buffer_},
                 metal_kernel_arg{param_buffer_}};

    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(relu,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::relu_image_compute,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();

REGISTER_LITE_KERNEL(relu6,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::relu6_image_compute,
                     def)
.BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
.BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kMetalTexture2DArray))})
.Finalize();


REGISTER_LITE_KERNEL(relu,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::relu_image_compute_half,
                     def)
.BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
.BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                          PRECISION(kFP16),
                                          DATALAYOUT(kMetalTexture2DArray))})
.Finalize();

REGISTER_LITE_KERNEL(relu6,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::relu6_image_compute_half,
                     def)
.BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
.BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                          PRECISION(kFP16),
                                          DATALAYOUT(kMetalTexture2DArray))})
.Finalize();