// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/x86/pow_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

void PowCompute::Run() {
  LOG(INFO) << "PowCompute";
  auto& param = Param<operators::PowParam>();
  const float* x_data = param.X->data<float>();
  float* output_data = param.Out->mutable_data<float>();
  DDim x_dims = param.X->dims();
  float scale = 1.0;
  float shift = 0.0;
  float power = param.factor;

  lite::x86::math::power(
      x_data, output_data, x_dims.production(), scale, shift, power);
}
}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    pow, kX86, kFloat, kNCHW, paddle::lite::kernels::x86::PowCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
