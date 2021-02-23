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

#include "lite/kernels/metal/image_op/nearest_interp_image_compute.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/metal/image_op/metal_params.h"
#include "lite/backends/metal/metal_debug.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void NearestInterpImageCompute::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();

  input_buffer_ = param.X->data<float, MetalImage>();
  output_buffer_ = param.Out->mutable_data<float, MetalImage>(
      output_dims, input_buffer_->transpose_);

  int input_h = static_cast<int>(input_buffer_->pad_to_four_dim_[2]);
  int input_w = static_cast<int>(input_buffer_->pad_to_four_dim_[3]);
  int output_h = static_cast<int>(output_buffer_->pad_to_four_dim_[2]);
  int output_w = static_cast<int>(output_buffer_->pad_to_four_dim_[3]);

  float ratio_w = 1.0f;
  float ratio_h = 1.0f;
  float align_delta = 0.0f;
  if (param.align_corners) {
      ratio_w = (float (input_w) - 1.0f) / (float(output_w) - 1.0f);
      ratio_h = (float (input_h) - 1.0f) / (float(output_h) - 1.0f);
      align_delta = 0.5f;
  } else {
      ratio_w = float (input_w) / float(output_w);
      ratio_h = float (input_h) / float(output_h);
      align_delta = 0.0;
  }

  NearestInterpMetalParam metal_param{ratio_h, ratio_w, align_delta};

  param_buffer_ = mtl_ctx->CreateBuffer(*device,
                                        &metal_param,
                                        sizeof(metal_param),
                                        METAL_ACCESS_FLAG::CPUWriteOnly);

  string function_name = "nearest_interp";
  kernel_ = mtl_ctx->GetKernel(*device, function_name);
}

void NearestInterpImageCompute::Run() {
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

    auto args = {MetalKernelArgument(input_buffer_),
                 MetalKernelArgument(output_buffer_),
                 MetalKernelArgument(param_buffer_)};

    kernel_->Execute(*queue, global_work_size, false, args);
    queue->WaitUntilComplete();
  }

#if LITE_METAL_SAVE_TENSOR
  MetalDebug::SaveOutput("nearest_interp", output_buffer_);
#endif
}

void NearestInterpImageComputeHalf::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();

  input_buffer_ = param.X->data<MetalHalf, MetalImage>();
  output_buffer_ = param.Out->mutable_data<MetalHalf, MetalImage>(
      output_dims, input_buffer_->transpose_);

  int input_h = static_cast<int>(input_buffer_->pad_to_four_dim_[2]);
  int input_w = static_cast<int>(input_buffer_->pad_to_four_dim_[3]);
  int output_h = static_cast<int>(output_buffer_->pad_to_four_dim_[2]);
  int output_w = static_cast<int>(output_buffer_->pad_to_four_dim_[3]);

  float ratio_w = 1.0f;
  float ratio_h = 1.0f;
  float align_delta = 0.0f;
  if (param.align_corners) {
      ratio_w = (float (input_w) - 1.0f) / (float(output_w) - 1.0f);
      ratio_h = (float (input_h) - 1.0f) / (float(output_h) - 1.0f);
      align_delta = 0.5f;
  } else {
      ratio_w = float (input_w) / float(output_w);
      ratio_h = float (input_h) / float(output_h);
      align_delta = 0.0;
  }


  BilinearInterPMetalParam metal_param{ratio_h, ratio_w, align_delta};

  param_buffer_ = mtl_ctx->CreateBuffer(*device,
                                        &metal_param,
                                        sizeof(metal_param),
                                        METAL_ACCESS_FLAG::CPUWriteOnly);

  string function_name = "nearest_interp_half";
  kernel_ = mtl_ctx->GetKernel(*device, function_name);
}

void NearestInterpImageComputeHalf::Run() {
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

    auto args = {MetalKernelArgument(input_buffer_),
                 MetalKernelArgument(output_buffer_),
                 MetalKernelArgument(param_buffer_)};

    kernel_->Execute(*queue, global_work_size, false, args);
    queue->WaitUntilComplete();
  }

#if LITE_METAL_SAVE_TENSOR
  MetalDebug::SaveOutput("nearest_interp", output_buffer_);
#endif
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(nearest_interp,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::NearestInterpImageCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Scale",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();

REGISTER_LITE_KERNEL(
    nearest_interp,
    kMetal,
    kFP16,
    kMetalTexture2DArray,
    paddle::lite::kernels::metal::NearestInterpImageComputeHalf,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMetal),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Scale",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();