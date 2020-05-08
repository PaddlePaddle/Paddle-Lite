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

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    mean, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::MeanCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
