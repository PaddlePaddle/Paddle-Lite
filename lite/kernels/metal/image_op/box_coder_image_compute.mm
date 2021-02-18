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
#include <algorithm>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/backends/metal/metal_debug.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void BoxCoderImageCompute::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.proposals->dims();
  auto prior_box_dims = param.prior_box->dims();

  assert(param.code_type == "decode_center_size" && param.box_normalized == true);

  prior_box_buffer_ = param.prior_box->data<float, MetalImage>();
  prior_box_var_buffer_ = param.prior_box_var->data<float, MetalImage>();
  target_box_buffer_ = param.target_box->data<float, MetalImage>();
  output_buffer_ = param.proposals->mutable_data<float, MetalImage>(output_dims);

  std::string function_name = "boxcoder_float";
  kernel_ = mtl_ctx->GetKernel(*device, function_name);
}

void BoxCoderImageCompute::Run() {
  auto output_width = output_buffer_->texture_width_;
  auto output_height = output_buffer_->texture_height_;
  auto output_array_length = output_buffer_->array_length_;

  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto mtl_dev = mtl_ctx->GetDefaultDevice();

  {
    auto queue = mtl_ctx->GetDefaultQueue(*mtl_dev);
    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                   static_cast<MetalUint>(output_height),
                                   static_cast<MetalUint>(output_array_length)};

    auto args = {MetalKernelArgument{prior_box_buffer_},
                 MetalKernelArgument{prior_box_var_buffer_},
                 MetalKernelArgument{target_box_buffer_},
                 MetalKernelArgument{output_buffer_}};
    kernel_->Execute(*queue, global_work_size, false, args);
    queue->WaitUntilComplete();
  }

#if LITE_METAL_SAVE_TENSOR
  MetalDebug::SaveOutput("transpose", output_buffer_);
#endif
}

void BoxCoderImageComputeHalf::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.proposals->dims();
  auto prior_box_dims = param.prior_box->dims();

  prior_box_buffer_ = param.prior_box->data<MetalHalf, MetalImage>();
  prior_box_var_buffer_ = param.prior_box_var->data<MetalHalf, MetalImage>();
  target_box_buffer_ = param.target_box->data<MetalHalf, MetalImage>();
  output_buffer_ = param.proposals->mutable_data<MetalHalf, MetalImage>(output_dims);

  std::string function_name = "boxcoder_half";
  kernel_ = mtl_ctx->GetKernel(*device, function_name);
}

void BoxCoderImageComputeHalf::Run() {
  auto output_width = output_buffer_->texture_width_;
  auto output_height = output_buffer_->texture_height_;
  auto output_array_length = output_buffer_->array_length_;

  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto mtl_dev = mtl_ctx->GetDefaultDevice();

  {
    auto queue = mtl_ctx->GetDefaultQueue(*mtl_dev);
    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                   static_cast<MetalUint>(output_height),
                                   static_cast<MetalUint>(output_array_length)};

    auto args = {MetalKernelArgument{prior_box_buffer_},
                 MetalKernelArgument{prior_box_var_buffer_},
                 MetalKernelArgument{target_box_buffer_},
                 MetalKernelArgument{output_buffer_}};

    kernel_->Execute(*queue, global_work_size, false, args);
    queue->WaitUntilComplete();
  }

#if LITE_METAL_SAVE_TENSOR
  MetalDebug::SaveOutput("transpose", output_buffer_);
#endif
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(box_coder,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::BoxCoderImageCompute,
                     def)
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

REGISTER_LITE_KERNEL(box_coder,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::BoxCoderImageComputeHalf,
                     def)
        .BindInput("PriorBox",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kMetalTexture2DArray))})
        .BindInput("PriorBoxVar",
                   {LiteType::GetTensorTy(TARGET(kMetal),
                                                         PRECISION(kFP16),
                                                         DATALAYOUT(kMetalTexture2DArray))})
        .BindInput("TargetBox",
                   {LiteType::GetTensorTy(TARGET(kMetal),
                                          PRECISION(kFP16),
                                          DATALAYOUT(kMetalTexture2DArray))})
        .BindOutput("OutputBox",
                    {LiteType::GetTensorTy(TARGET(kMetal),
                                           PRECISION(kFP16),
                                           DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();
