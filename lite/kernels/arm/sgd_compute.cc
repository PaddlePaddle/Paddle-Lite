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

#include "lite/kernels/arm/sgd_compute.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void SGDCompute::Run() {
  auto& param = this->Param<param_t>();
  const auto* parameter = param.Param;
  const auto* grad = param.Grad;
  const auto* lr_tensor = param.LearningRate;
  auto* parameter_output = param.ParamOut;

  auto dims = parameter->dims();
  auto parameter_data = parameter->data<float>();
  auto grad_data = grad->data<float>();
  auto lr = *(lr_tensor->data<float>());
  auto parameter_out_data = parameter_output->mutable_data<float>();

  int element_num = dims.production();
#pragma omp parallel for
  for (int i = 0; i < element_num; i++) {
    parameter_out_data[i] = parameter_data[i] - lr * grad_data[i];
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    sgd, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::SGDCompute, def)
    .BindInput("Param", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Grad", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("LearningRate", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("ParamOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
