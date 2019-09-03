// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/axpy_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void AxpyCompute::Run() {
  auto& param = Param<operators::AxpyParam>();
  lite::Tensor* scale = param.Scale;
  lite::Tensor* x = param.X;
  lite::Tensor* bias = param.Bias;
  lite::Tensor* out = param.Out;

  const float* scale_ptr = scale->data<float>();
  const float* x_ptr = x->data<float>();
  const float* bias_ptr = bias->data<float>();
  float* out_ptr = out->mutable_data<float>();

  auto bias_dims = bias->dims();
  int num = bias_dims[0];
  int channel = bias_dims[1];
  int size = bias_dims[2] * bias_dims[3];
  int in_channel = channel * size;

  lite::arm::math::axpy_kernel_fp32(
      scale_ptr, x_ptr, bias_ptr, out_ptr, num, channel, size, in_channel);
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    axpy, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::AxpyCompute, def)
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
