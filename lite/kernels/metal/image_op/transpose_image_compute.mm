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

#include "lite/kernels/metal/image_op/transpose_image_compute.h"
#include <algorithm>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/backends/metal/metal_debug.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void TransposeImageCompute::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  auto input_dims = param.x->dims();

  input_buffer_ = param.x->data<float, MetalImage>();
  output_buffer_ = param.output->mutable_data<float, MetalImage>(output_dims);

  std::vector<int> expected_transpose = {0, 2, 3, 1};

  if (input_buffer_->transpose_ == expected_transpose) {
    throw std::logic_error("expected transpose is not equal with input_buffer");
  }

  auto rank = input_buffer_->tensor_dim_.size();
  std::vector<int> axis = {0, 1, 2, 3};
  for (int i = 0; i < param.axis.size(); i++) {
    axis[4 - rank + i] = static_cast<int>(4 - rank + (int)(param.axis.size()));
  }

  std::vector<int> trans_axis = {axis[expected_transpose[0]],
                                 axis[expected_transpose[1]],
                                 axis[expected_transpose[2]],
                                 axis[expected_transpose[3]]};

  std::vector<int> naxis = {0, 0, 0, 0};
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (input_buffer_->transpose_[j] == trans_axis[i]) {
        naxis[i] = j;
        break;
      }
    }
  }

  TransposeMetalParam transpose_params = {
      static_cast<int>(input_dims[3]),
      static_cast<int>(output_dims[3]),
      {naxis[0], naxis[1], naxis[2], naxis[3]}};

  param_buffer_ = mtl_ctx->CreateBuffer(*device,
                                        &transpose_params,
                                        sizeof(transpose_params),
                                        METAL_ACCESS_FLAG::CPUWriteOnly);

  std::string function_name = "transpose_" + std::to_string(rank);
  kernel_ = mtl_ctx->GetKernel(*device, function_name);
}

void TransposeImageCompute::Run() {
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

    auto args = {MetalKernelArgument{input_buffer_},
                 MetalKernelArgument{output_buffer_},
                 MetalKernelArgument{param_buffer_}};
    kernel_->Execute(*queue, global_work_size, false, args);
    queue->WaitUntilComplete();
  }

#if LITE_METAL_SAVE_TENSOR
  MetalDebug::SaveOutput("transpose", output_buffer_);
#endif
}

void TransposeImageComputeHalf::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.output->dims();
  auto input_dims = param.x->dims();

  input_buffer_ = param.x->data<MetalHalf, MetalImage>();
  output_buffer_ =
      param.output->mutable_data<MetalHalf, MetalImage>(output_dims);

  std::vector<int> expected_transpose = {0, 2, 3, 1};

  if (input_buffer_->transpose_ == expected_transpose) {
    throw std::logic_error("expected transpose is not equal with input_buffer");
  }

  auto rank = input_buffer_->tensor_dim_.size();
  std::vector<int> axis = {0, 1, 2, 3};
  for (int i = 0; i < param.axis.size(); i++) {
    axis[4 - rank + i] = static_cast<int>(4 - rank + (int)(param.axis.size()));
  }

  std::vector<int> trans_axis = {axis[expected_transpose[0]],
                                 axis[expected_transpose[1]],
                                 axis[expected_transpose[2]],
                                 axis[expected_transpose[3]]};

  std::vector<int> naxis = {0, 0, 0, 0};
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (input_buffer_->transpose_[j] == trans_axis[i]) {
        naxis[i] = j;
        break;
      }
    }
  }

  TransposeMetalParam transpose_params = {
      static_cast<int>(input_dims[3]),
      static_cast<int>(output_dims[3]),
      {naxis[0], naxis[1], naxis[2], naxis[3]}};

  param_buffer_ = mtl_ctx->CreateBuffer(*device,
                                        &transpose_params,
                                        sizeof(transpose_params),
                                        METAL_ACCESS_FLAG::CPUWriteOnly);

  std::string function_name = "transpose_" + std::to_string(rank) + "_half";
  kernel_ = mtl_ctx->GetKernel(*device, function_name);
}

void TransposeImageComputeHalf::Run() {
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

    auto args = {MetalKernelArgument{input_buffer_},
                 MetalKernelArgument{output_buffer_},
                 MetalKernelArgument{param_buffer_}};
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

REGISTER_LITE_KERNEL(transpose,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::TransposeImageCompute,
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

REGISTER_LITE_KERNEL(transpose,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::TransposeImageComputeHalf,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose2,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::TransposeImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose2,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::TransposeImageComputeHalf,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();
