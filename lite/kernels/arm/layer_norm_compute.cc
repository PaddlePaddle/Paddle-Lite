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

#include "lite/kernels/arm/layer_norm_compute.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void LayerNormCompute::PrepareForRun() {}

void LayerNormCompute::Run() {
  auto& param = this->Param<operators::LayerNormParam>();

  auto input_dims = param.X->dims();

  const auto* x_data = param.X->data<float>();
  const auto* scale = param.Scale ? param.Scale->data<float>() : nullptr;
  const auto* bias = param.Bias ? param.Bias->data<float>() : nullptr;
  auto* o_data = param.Y->mutable_data<float>();
  auto* mean = param.Mean->mutable_data<float>();
  auto* var = param.Variance->mutable_data<float>();

  int axis = param.begin_norm_axis;
  auto matrix_dim = param.X->dims().Flatten2D(axis);
  int left = matrix_dim[0];
  int right = matrix_dim[1];

  lite::arm::math::matrix_norm_row(
      x_data, scale, bias, o_data, mean, var, param.epsilon, left, right);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(layer_norm,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::LayerNormCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Mean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Variance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
