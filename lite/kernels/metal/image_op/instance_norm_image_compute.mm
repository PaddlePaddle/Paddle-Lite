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

#include "lite/kernels/metal/image_op/instance_norm_image_compute.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/metal/image_op/metal_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

#define ALIGEN_C4_SIZE(n, c, h, w) ((n * c + 3) / 4 * h * w) * 4

void InstanceNormImageCompute::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

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

  output_buffer_ = param.out->mutable_data<float, MetalImage>(param.out->dims());
  input_buffer_ = param.x->data<float, MetalImage>();

  uint16_t has_relu = (uint16_t)param.fuse_relu;
  InstanceNormReluMetalParam metal_param{has_relu};
  params_buffer_ = mtl_ctx->CreateBuffer(
      *device, &metal_param, sizeof(metal_param), METAL_ACCESS_FLAG::CPUWriteOnly);

  auto bias_raw_buffer = param.bias->data<float>();
  auto scale_raw_buffer = param.scale->data<float>();
  auto mean_raw_buffer = param.saved_mean->data<float>();
  auto variance_ptr = param.saved_variance->data<float>();

  auto bias_host_ptr = const_cast<float*>(bias_raw_buffer);
  auto scale_host_ptr = const_cast<float*>(scale_raw_buffer);

  auto scale_size = ALIGEN_C4_SIZE(output_tensor_n_, scale_dims[0], 1, 1);
  auto bias_size = ALIGEN_C4_SIZE(output_tensor_n_, bias_dims[0], 1, 1);

  auto count = scale_dims.production();

  scale_buffer_ = mtl_ctx->CreateBuffer(*device, scale_size * sizeof(float));

  bias_buffer_ = mtl_ctx->CreateBuffer(*device, bias_size * sizeof(float));
  auto bias_dev_ptr = (float*)(bias_buffer_->buffer().contents);
  auto scale_dev_ptr = (float*)(scale_buffer_->buffer().contents);

  for (int i = 0; i < count; i++) {
    auto inv_std = 1.0f / std::sqrt(variance_ptr[i] + param.epsilon);
    bias_dev_ptr[i] = bias_host_ptr[i] - mean_raw_buffer[i] * inv_std * scale_host_ptr[i];
    scale_dev_ptr[i] = inv_std * scale_host_ptr[i];
  }

  for (int i = 1; i < (scale_size / scale_dims[0]); i++) {
    memcpy(bias_dev_ptr + i * scale_dims[0], bias_dev_ptr, count * sizeof(float));
    memcpy(scale_dev_ptr + i * scale_dims[0], scale_dev_ptr, count * sizeof(float));
  }
}

void InstanceNormImageCompute::Run() {
  const auto& param = this->Param<param_t>();
  auto input_dims = param.x->dims();
  auto output_dims = param.out->dims();
  auto output_width = output_dims[3];
  auto output_height = output_dims[2];
  auto output_array_length = (output_dims[0] * output_dims[1] + 3) / 4;

  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto mtl_dev = mtl_ctx->GetDefaultDevice();

  {
    std::string function_name = "instance_norm_relu";
    auto queue = mtl_ctx->GetDefaultQueue(*mtl_dev);
    auto kernel = mtl_ctx->GetKernel(*mtl_dev, function_name);

    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                    static_cast<MetalUint>(output_height),
                                    static_cast<MetalUint>(output_array_length)};

    auto args = {MetalKernelArgument(input_buffer_),
                 MetalKernelArgument(output_buffer_),
                 MetalKernelArgument(params_buffer_)};

    kernel->Execute(*queue, global_work_size, false, args);
    queue->WaitUntilComplete();
  }
}

void InstanceNormImageComputeHalf::PrepareForRun() {
  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto device = mtl_ctx->GetDefaultDevice();

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

  output_buffer_ = param.out->mutable_data<MetalHalf, MetalImage>(param.out->dims());
  input_buffer_ = param.x->data<MetalHalf, MetalImage>();

  uint16_t has_relu = (uint16_t)param.fuse_relu;
  InstanceNormReluMetalParam metal_param{has_relu};

  auto bias_raw_buffer = param.bias->data<float>();
  auto scale_raw_buffer = param.scale->data<float>();
  auto mean_raw_buffer = param.saved_mean->data<float>();
  auto variance_ptr = param.saved_variance->data<float>();

  auto bias_host_ptr = const_cast<float*>(bias_raw_buffer);
  auto scale_host_ptr = const_cast<float*>(scale_raw_buffer);

  auto scale_size = ALIGEN_C4_SIZE(output_tensor_n_, scale_dims[0], 1, 1);
  auto bias_size = ALIGEN_C4_SIZE(output_tensor_n_, bias_dims[0], 1, 1);

  auto count = scale_dims.production();

  scale_buffer_ = mtl_ctx->CreateBuffer(*device, scale_size * sizeof(MetalHalf));
  bias_buffer_ = mtl_ctx->CreateBuffer(*device, bias_size * sizeof(MetalHalf));
  auto bias_dev_ptr = (MetalHalf*)(bias_buffer_->buffer().contents);
  auto scale_dev_ptr = (MetalHalf*)(scale_buffer_->buffer().contents);

  for (int i = 0; i < count; i++) {
    auto inv_std = 1.0f / std::sqrt(variance_ptr[i] + param.epsilon);
    bias_dev_ptr[i] =
        MetalFloat2Half(bias_host_ptr[i] - mean_raw_buffer[i] * inv_std * scale_host_ptr[i]);
    scale_dev_ptr[i] = MetalFloat2Half(inv_std * scale_host_ptr[i]);
  }

  for (int i = 1; i < (scale_size / scale_dims[0]); i++) {
    memcpy(bias_dev_ptr + i * scale_dims[0], bias_dev_ptr, count * sizeof(MetalHalf));
    memcpy(scale_dev_ptr + i * scale_dims[0], scale_dev_ptr, count * sizeof(MetalHalf));
  }
}

void InstanceNormImageComputeHalf::Run() {
  const auto& param = this->Param<param_t>();
  auto input_dims = param.x->dims();
  auto output_dims = param.out->dims();
  auto output_width = output_dims[3];
  auto output_height = output_dims[2];
  auto output_array_length = (output_dims[0] * output_dims[1] + 3) / 4;

  auto& context = ctx_->As<ContextMetal>();
  auto mtl_ctx = (MetalContext*)context.context();
  auto mtl_dev = mtl_ctx->GetDefaultDevice();

  {
    std::string function_name = "instance_norm_relu_half";
    auto queue = mtl_ctx->GetDefaultQueue(*mtl_dev);
    auto kernel = mtl_ctx->GetKernel(*mtl_dev, function_name);

    MetalUint3 global_work_size = {static_cast<MetalUint>(output_width),
                                    static_cast<MetalUint>(output_height),
                                    static_cast<MetalUint>(output_array_length)};

    auto args = {MetalKernelArgument(input_buffer_),
                 MetalKernelArgument(output_buffer_),
                 MetalKernelArgument(params_buffer_)};
    // TODO: (lzy) execute method
    kernel->Execute(*queue, global_work_size, false, args);
    queue->WaitUntilComplete();
  }
}

}  // namespace metal
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(instance_norm,
                     kMetal,
                     kFloat,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::InstanceNormImageCompute,
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
        .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kHost),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kNCHW))})
        .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kHost),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kNCHW))})
        .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kMetal),
                                                     PRECISION(kFloat),
                                                     DATALAYOUT(kMetalTexture2DArray))})
        .Finalize();


REGISTER_LITE_KERNEL(instance_norm,
                     kMetal,
                     kFP16,
                     kMetalTexture2DArray,
                     paddle::lite::kernels::metal::InstanceNormImageComputeHalf,
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
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kHost),
                                                    PRECISION(kFloat),
                                                    DATALAYOUT(kNCHW))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kHost),
                                                        PRECISION(kFloat),
                                                        DATALAYOUT(kNCHW))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kMetal),
                                            PRECISION(kFP16),
                                            DATALAYOUT(kMetalTexture2DArray))})
    .Finalize();