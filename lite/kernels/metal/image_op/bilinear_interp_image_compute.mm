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
#include "bilinear_interp_image_compute.h"
#include "lite/kernels/metal/image_op/metal_params.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

void bilinear_interp_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();

  input_buffer_ = param.X->data<float, metal_image>();
  output_buffer_ =
      param.Out->mutable_data<float, metal_image>(output_dims, input_buffer_->transpose_);

  int input_h = static_cast<int>(input_buffer_->padToFourDim_[2]);
  int input_w = static_cast<int>(input_buffer_->padToFourDim_[3]);
  int output_h = static_cast<int>(output_buffer_->padToFourDim_[2]);
  int output_w = static_cast<int>(output_buffer_->padToFourDim_[3]);

  float delta_h = 0;
  float delta_w = 0;

  // 根据align_corners与图像宽高，决定是否要与左上角的像素对齐（像素减1）
  if (param.align_corners && output_h > 1) {
    delta_h = 1.0;
  }
  if (param.align_corners && output_w > 1) {
    delta_w = 1.0;
  }
  float ratio_h = ((float)(input_h)-delta_h) / ((float)(output_h)-delta_h);
  float ratio_w = ((float)(input_w)-delta_w) / ((float)(output_w)-delta_w);

  float align_delta = 0;
  bool align_flag = (param.align_mode == 0 && !param.align_corners);

  // 在metal kernel中，align_delta直接参与计算，若align_delta = 0.0，不进行中心对齐；
  // 若param.align_mode == 0 且param.align_corners == false，则align_delta = 0.5，进行中心对齐
  if (align_flag) {
    align_delta = 0.5;
  }

  BilinearInterPMetalParam metal_param{ratio_h, ratio_w, align_delta};

  param_buffer_ = mtl_ctx->create_buffer(
      *device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);

  string function_name = "bilinear_interp_float";
  kernel_ = mtl_ctx->get_kernel(*device, function_name);
}

void bilinear_interp_image_compute::Run() {
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

    auto args = {metal_kernel_arg(input_buffer_),
                 metal_kernel_arg(output_buffer_),
                 metal_kernel_arg(param_buffer_)};

    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

void bilinear_interp_image_compute_half::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.Out->dims();

  input_buffer_ = param.X->data<metal_half, metal_image>();
  output_buffer_ =
      param.Out->mutable_data<metal_half, metal_image>(output_dims, input_buffer_->transpose_);

  int input_h = static_cast<int>(input_buffer_->padToFourDim_[2]);
  int input_w = static_cast<int>(input_buffer_->padToFourDim_[3]);
  int output_h = static_cast<int>(output_buffer_->padToFourDim_[2]);
  int output_w = static_cast<int>(output_buffer_->padToFourDim_[3]);

  float delta_h = 0;
  float delta_w = 0;

  // 根据align_corners与图像宽高，决定是否要与左上角的像素对齐（像素减1）
  if (param.align_corners && output_h > 1) {
    delta_h = 1.0;
  }
  if (param.align_corners && output_w > 1) {
    delta_w = 1.0;
  }
  float ratio_h = ((float)(input_h)-delta_h) / ((float)(output_h)-delta_h);
  float ratio_w = ((float)(input_w)-delta_w) / ((float)(output_w)-delta_w);

  float align_delta = 0;
  bool align_flag = (param.align_mode == 0 && !param.align_corners);

  // 在metal kernel中，align_delta直接参与计算，若align_delta = 0.0，不进行中心对齐；
  // 若param.align_mode == 0 且param.align_corners == false，则align_delta = 0.5，进行中心对齐
  if (align_flag) {
    align_delta = 0.5;
  }

  BilinearInterPMetalParam metal_param{ratio_h, ratio_w, align_delta};

  param_buffer_ = mtl_ctx->create_buffer(
      *device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);

  string function_name = "bilinear_interp_half";
  kernel_ = mtl_ctx->get_kernel(*device, function_name);
}

void bilinear_interp_image_compute_half::Run() {
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

    auto args = {metal_kernel_arg(input_buffer_),
                 metal_kernel_arg(output_buffer_),
                 metal_kernel_arg(param_buffer_)};

    kernel_->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(bilinear_interp,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::bilinear_interp_image_compute,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindInput("OutSize", {LiteType::GetTensorTy(TARGET(kHost),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kNCHW))})
        .BindInput("SizeTensor", {LiteType::GetTensorTy(TARGET(kHost),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kNCHW))})
        .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost),
                                                    PRECISION(kFloat),
                                                   DATALAYOUT(kNCHW))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();


REGISTER_LITE_KERNEL(bilinear_interp,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::bilinear_interp_image_compute_half,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kMetalTexture2DArray))})
        .BindInput("OutSize", {LiteType::GetTensorTy(TARGET(kHost),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kNCHW))})
        .BindInput("SizeTensor", {LiteType::GetTensorTy(TARGET(kHost),
                                                        PRECISION(kFloat),
                                                        DATALAYOUT(kNCHW))})
        .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kNCHW))})
        .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kMetal),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();