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

#include "lite/kernels/metal/image_op/batch_norm_image_compute.h"
#include "lite/core/op_registry.h"

using namespace std;

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

#define ALIGEN_C4_SIZE(n, c, h, w) ((n * c + 3) / 4 * h * w) * 4

void batch_norm_image_compute::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.x->dims();
  auto input_dims = param.x->dims();
  auto scale_dims = param.scale->dims();
  auto bias_dims = param.bias->dims();

  input_tensor_n_ = static_cast<int>(input_dims[0]);
  input_tensor_c_ = static_cast<int>(input_dims[1]);
  input_tensor_h_ = static_cast<int>(input_dims[2]);
  input_tensor_w_ = static_cast<int>(input_dims[3]);

  output_tensor_n_ = static_cast<int>(output_dims[0]);
  output_tensor_c_ = static_cast<int>(output_dims[1]);
  output_tensor_h_ = static_cast<int>(output_dims[2]);
  output_tensor_w_ = static_cast<int>(output_dims[3]);

  output_buffer_ = param.y->mutable_data<float, metal_image>(param.y->dims());
  input_buffer_ = param.x->data<float, metal_image>();

  auto bias_raw_buffer = param.bias->data<float>();
  auto scale_raw_buffer = param.scale->data<float>();
  auto mean_raw_buffer = param.mean->data<float>();
  auto variance_ptr = param.variance->data<float>();

  auto bias_host_ptr = const_cast<float*>(bias_raw_buffer);
  auto scale_host_ptr = const_cast<float*>(scale_raw_buffer);

  auto scale_size = ALIGEN_C4_SIZE(output_tensor_n_, scale_dims[0], 1, 1);
  auto bias_size = ALIGEN_C4_SIZE(output_tensor_n_, bias_dims[0], 1, 1);

  auto count = scale_dims.production();

  scale_buffer_ = mtl_ctx->create_buffer(*device, scale_size * sizeof(float));

  bias_buffer_ = mtl_ctx->create_buffer(*device, bias_size * sizeof(float));
  auto bias_dev_ptr = (float*)(bias_buffer_->get_buffer().contents);
  auto scale_dev_ptr = (float*)(scale_buffer_->get_buffer().contents);

  for (int i = 0; i < count; i++) {
    auto invStd = 1.0f / std::sqrt(variance_ptr[i] + param.epsilon);
    bias_dev_ptr[i] = bias_host_ptr[i] - mean_raw_buffer[i] * invStd * scale_host_ptr[i];
    scale_dev_ptr[i] = invStd * scale_host_ptr[i];
  }

  for (int i = 1; i < (scale_size / scale_dims[0]); i++) {
    memcpy(bias_dev_ptr + i * scale_dims[0], bias_dev_ptr, count * sizeof(float));
    memcpy(scale_dev_ptr + i * scale_dims[0], scale_dev_ptr, count * sizeof(float));
  }
}

void batch_norm_image_compute::Run() {
  const auto& param = this->Param<param_t>();
  auto input_dims = param.x->dims();
  auto output_dims = param.y->dims();
  auto output_width = output_dims[3];
  auto output_height = output_dims[2];
  auto output_array_length = (output_dims[0] * output_dims[1] + 3) / 4;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    string function_name = "batchnorm";
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    auto kernel = mtl_ctx->get_kernel(*mtl_dev, function_name);

    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    auto args = {metal_kernel_arg(input_buffer_),
                 metal_kernel_arg(output_buffer_),
                 metal_kernel_arg(scale_buffer_),
                 metal_kernel_arg(bias_buffer_)};

    kernel->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

void batch_norm_image_compute_half::PrepareForRun() {
  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto device = mtl_ctx->get_default_device();

  const auto& param = this->Param<param_t>();
  auto output_dims = param.x->dims();
  auto input_dims = param.x->dims();
  auto scale_dims = param.scale->dims();
  auto bias_dims = param.bias->dims();

  input_tensor_n_ = static_cast<int>(input_dims[0]);
  input_tensor_c_ = static_cast<int>(input_dims[1]);
  input_tensor_h_ = static_cast<int>(input_dims[2]);
  input_tensor_w_ = static_cast<int>(input_dims[3]);

  output_tensor_n_ = static_cast<int>(output_dims[0]);
  output_tensor_c_ = static_cast<int>(output_dims[1]);
  output_tensor_h_ = static_cast<int>(output_dims[2]);
  output_tensor_w_ = static_cast<int>(output_dims[3]);

  output_buffer_ = param.y->mutable_data<metal_half, metal_image>(param.y->dims());
  input_buffer_ = param.x->data<metal_half, metal_image>();

  auto bias_raw_buffer = param.bias->data<float>();
  auto scale_raw_buffer = param.scale->data<float>();
  auto mean_raw_buffer = param.mean->data<float>();
  auto variance_ptr = param.variance->data<float>();

  auto bias_host_ptr = const_cast<float*>(bias_raw_buffer);
  auto scale_host_ptr = const_cast<float*>(scale_raw_buffer);

  auto scale_size = ALIGEN_C4_SIZE(output_tensor_n_, scale_dims[0], 1, 1);
  auto bias_size = ALIGEN_C4_SIZE(output_tensor_n_, bias_dims[0], 1, 1);

  auto count = scale_dims.production();

  scale_buffer_ = mtl_ctx->create_buffer(*device, scale_size * sizeof(metal_half));
  bias_buffer_ = mtl_ctx->create_buffer(*device, bias_size * sizeof(metal_half));
  auto bias_dev_ptr = (metal_half*)(bias_buffer_->get_buffer().contents);
  auto scale_dev_ptr = (metal_half*)(scale_buffer_->get_buffer().contents);

  for (int i = 0; i < count; i++) {
    auto invStd = 1.0f / std::sqrt(variance_ptr[i] + param.epsilon);
    bias_dev_ptr[i] =
        MetalFloat2Half(bias_host_ptr[i] - mean_raw_buffer[i] * invStd * scale_host_ptr[i]);
    scale_dev_ptr[i] = MetalFloat2Half(invStd * scale_host_ptr[i]);
  }

  for (int i = 1; i < (scale_size / scale_dims[0]); i++) {
    memcpy(bias_dev_ptr + i * scale_dims[0], bias_dev_ptr, count * sizeof(metal_half));
    memcpy(scale_dev_ptr + i * scale_dims[0], scale_dev_ptr, count * sizeof(metal_half));
  }
}

void batch_norm_image_compute_half::Run() {
  const auto& param = this->Param<param_t>();
  auto input_dims = param.x->dims();
  auto output_dims = param.y->dims();
  auto output_width = output_dims[3];
  auto output_height = output_dims[2];
  auto output_array_length = (output_dims[0] * output_dims[1] + 3) / 4;

  auto& context = ctx_->As<MetalContext>();
  auto mtl_ctx = (metal_context*)context.context();
  auto mtl_dev = mtl_ctx->get_default_device();

  {
    string function_name = "batchnorm_half";
    auto queue = mtl_ctx->get_default_queue(*mtl_dev);
    auto kernel = mtl_ctx->get_kernel(*mtl_dev, function_name);

    metal_uint3 global_work_size = {static_cast<metal_uint>(output_width),
                                    static_cast<metal_uint>(output_height),
                                    static_cast<metal_uint>(output_array_length)};

    auto args = {metal_kernel_arg{input_buffer_},
                 metal_kernel_arg{output_buffer_},
                 metal_kernel_arg{scale_buffer_},
                 metal_kernel_arg{bias_buffer_}};

    kernel->execute(*queue, global_work_size, 0, args);
    queue->wait_until_complete();
  }
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(batch_norm,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::batch_norm_image_compute,
                     def)
        .BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kMetalTexture2DArray))})
        .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost),
                                               PRECISION(kFloat),
                                               DATALAYOUT(kNCHW))})
        .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost),
                                                PRECISION(kFloat),
                                                DATALAYOUT(kNCHW))})
        .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kHost),
                                                  PRECISION(kFloat),
                                                  DATALAYOUT(kNCHW))})
        .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kHost),
                                                   PRECISION(kFloat),
                                                   DATALAYOUT(kNCHW))})
        .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kHost),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kNCHW))})
        .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kHost),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kNCHW))})
        .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kHost),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kNCHW))})
        .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kHost),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kNCHW))})
        .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kMetal),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();


REGISTER_LITE_KERNEL(batch_norm,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::batch_norm_image_compute_half,
                     def)
.BindInput("X", {LiteType::GetTensorTy(TARGET(kMetal),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kMetalTexture2DArray))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kNCHW))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost),
                                               PRECISION(kFloat),
                                               DATALAYOUT(kNCHW))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kHost),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kNCHW))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kHost),
                                                  PRECISION(kFloat),
                                                  DATALAYOUT(kNCHW))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kHost),
                                                      PRECISION(kFloat),
                                                      DATALAYOUT(kNCHW))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kHost),
                                                    PRECISION(kFloat),
                                                    DATALAYOUT(kNCHW))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kHost),
                                                        PRECISION(kFloat),
                                                        DATALAYOUT(kNCHW))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kHost),
                                                  PRECISION(kFloat),
                                                  DATALAYOUT(kNCHW))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kMetal),
                                            PRECISION(kFP16),
                                            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();