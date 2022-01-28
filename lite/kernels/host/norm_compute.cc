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

#include "lite/kernels/host/norm_compute.h"
#include "lite/backends/host/math/norm.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void NormCompute::Run() {
  auto& param = this->Param<operators::NormParam>();

  auto input_dims = param.X->dims();
  int dim_size = param.X->dims().size();
  auto axis = (param.axis < 0) ? param.axis + dim_size : param.axis;

  const auto* x_data = param.X->data<float>();
  auto* o_data = param.Out->mutable_data<float>();
  int pre_n = input_dims.count(0, axis);
  int post_n = input_dims.count(axis + 1, dim_size);
  int n = input_dims[axis];
  lite::host::math::norm(x_data, pre_n, n, post_n, param.epsilon, o_data);
}

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
  if (param.asvector) {
    pre = 1;
    post = 1;
    n = xdims.count(0, xdims.size());
  }
  lite::host::math::p_norm(
      x_data, pre, n, post, param.epsilon, out_data, porder);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    norm, kHost, kFloat, kNCHW, paddle::lite::kernels::host::NormCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Norm", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

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
