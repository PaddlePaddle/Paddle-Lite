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

#include "lite/kernels/host/activation_grad_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void SquareGradCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto out_grad_dims = param.Out_grad->dims();
  auto out_grad_data = param.Out_grad->data<float>();

  auto x_data = param.X->data<float>();
  auto x_grad_data = param.X_grad->mutable_data<float>();
  for (int i = 0; i < out_grad_dims.production(); i++) {
    x_grad_data[i] = out_grad_data[i] * 2.0 * x_data[i];
  }
}

void ReluGradCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.X);
  auto out_grad_dims = param.Out_grad->dims();
  auto out_grad_data = param.Out_grad->data<float>();

  auto x_data = param.X->data<float>();
  auto x_grad_data = param.X_grad->mutable_data<float>();
  for (int i = 0; i < out_grad_dims.production(); i++) {
    x_grad_data[i] = x_data[i] > 0 ? out_grad_data[i] : 0.0;
  }
}

void TanhGradCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(param.Out);
  auto out_grad_dims = param.Out_grad->dims();
  auto out_grad_data = param.Out_grad->data<float>();

  auto out_data = param.Out->data<float>();
  auto x_grad_data = param.X_grad->mutable_data<float>();
  for (int i = 0; i < out_grad_dims.production(); i++) {
    x_grad_data[i] = out_grad_data[i] *
                     (static_cast<float>(1.0) - out_data[i] * out_data[i]);
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(square_grad,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SquareGradCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Out@GRAD", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("X@GRAD", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(relu_grad,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SquareGradCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Out@GRAD", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("X@GRAD", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(tanh_grad,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SquareGradCompute,
                     def)
    .BindInput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Out@GRAD", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("X@GRAD", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
