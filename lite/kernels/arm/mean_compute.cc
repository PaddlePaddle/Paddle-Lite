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

#include "lite/kernels/arm/mean_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void MeanCompute::Run() {
  auto& param = this->Param<operators::MeanParam>();
  const auto* input = param.X;
  auto* output = param.Out;
  auto x_dim = input->dims();
  auto x_data = input->data<float>();
  auto out_data = output->mutable_data<float>();

  int x_size = x_dim.production();
  float sum = 0;
  for (int i = 0; i < x_size; i++) {
    sum += x_data[i];
  }
  out_data[0] = sum / x_size;
}

#ifdef LITE_WITH_TRAIN
void MeanGradCompute::Run() {
  auto& param = this->Param<operators::MeanGradParam>();
  const auto* input = param.X;
  const auto* out_grad = param.Out_grad;
  auto* input_grad = param.X_grad;

  auto out_grad_data = out_grad->data<float>();
  auto input_data = input->data<float>();
  auto input_grad_data = input_grad->mutable_data<float>();

  int input_grad_size = input_grad->dims().production();

  // TODO(mapingshuo): use parallel methods to accelerate this for loop
  for (int i = 0; i < input_grad_size; i++) {
    input_grad_data[i] = out_grad_data[0] / input_grad_size;
  }
}
#endif
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    mean, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::MeanCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

#ifdef LITE_WITH_TRAIN
REGISTER_LITE_KERNEL(mean_grad,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::MeanGradCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Out@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("X@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
#endif
