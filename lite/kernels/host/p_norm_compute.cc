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

#include "lite/kernels/host/p_norm_compute.h"
#include <vector>
#include "lite/backends/host/math/norm.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void PNormCompute::Run() {
  auto& param = this->Param<operators::PNormParam>();
  auto x = param.X;
  auto xdims = x->dims();
  float porder = param.porder;
  int axis = param.axis;
  const auto* x_data = x->data<float>();
  auto* out_data = param.Out->mutable_data<float>();
  if (axis < 0) {
    axis += xdims.size();
  }
  int pre = xdims.count(0, axis);
  int post = xdims.count(axis + 1, xdims.size());
  int n = xdims[axis];
  lite::host::math::p_norm(
      x_data, pre, n, post, param.epsilon, out_data, porder);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(p_norm,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::PNormCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();
