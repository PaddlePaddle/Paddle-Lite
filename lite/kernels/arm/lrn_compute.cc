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

#include "lite/kernels/arm/lrn_compute.h"
#include <string>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void LrnCompute::Run() {
  auto& param = Param<operators::LrnParam>();
  const float* x_data = param.X->data<float>();
  float* out_data = param.Out->mutable_data<float>();
  auto x_dims = param.X->dims();
  CHECK_EQ(x_dims.size(), 4);
  int num = x_dims[0];
  int channel = x_dims[1];
  int h = x_dims[2];
  int w = x_dims[3];
  const int n = param.n;
  const float alpha = param.alpha;
  const float beta = param.beta;
  const float k = param.k;
  if (param.norm_region == "AcrossChannels") {
    lite::arm::math::compute_across_channels(
        x_data, out_data, num, channel, h, w, n, alpha, beta, k);
  } else {
    lite::arm::math::compute_within_channels(
        x_data, out_data, num, channel, h, w, n, alpha, beta, k);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    lrn, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::LrnCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("MidOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
