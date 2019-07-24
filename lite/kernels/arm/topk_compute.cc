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

#include "lite/kernels/arm/topk_compute.h"
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void TopkCompute::Run() {
  auto& param = Param<operators::TopkParam>();
  const float* x_data = param.X->data<float>();
  float* out_data = param.Out->mutable_data<float>();
  DDim x_dims = param.x->dims();
  int K = param.K;
  int dim_size = x_dims.size();
  int m = x_dims.production() / x_dims[dim_size - 1];
  int n = x_dims[dim_size - 1];
  lite::arm::math::topk(x_data, out_data, m, n, K);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    scale, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::ScaleCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
