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

#include "lite/kernels/arm/activation_compute.h"
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void ReluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_relu<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void LeakyReluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto alpha = param.Leaky_relu_alpha;
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_relu_neg<float>(
      x_data, output_data, x_dims.production(), alpha, ctx.threads());
}

void ReluClippedCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto coef = param.Relu_clipped_coef;
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_clipped_relu<float>(
      x_data, output_data, x_dims.production(), coef, ctx.threads());
}

void PReluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto channel_shared = param.Prelu_channel_shared;
  auto channel_slope = param.Prelu_channel_slope->data<float>();
  auto output_data = param.Out->mutable_data<float>();

  int outer_size = x_dims[0];
  int channel_size = x_dims[1];
  int inner_size = x_dims[2] * x_dims[3];
  lite::arm::math::act_prelu<float>(x_data,
                                    output_data,
                                    outer_size,
                                    channel_size,
                                    inner_size,
                                    channel_shared,
                                    channel_slope,
                                    ctx.threads());
}

void SigmoidCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_sigmoid<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void TanhCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_tanh<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void SwishCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto beta = param.Swish_beta;
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_swish<float>(
      x_data, output_data, x_dims.production(), beta, ctx.threads());
}

void Relu6Compute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  float coef = 6.;
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_clipped_relu<float>(
      x_data, output_data, x_dims.production(), coef, ctx.threads());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    relu, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::ReluCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(leaky_relu,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::LeakyReluCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("alpha", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(relu_clipped,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ReluClippedCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Relu_clipped_coef", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    prelu, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::PReluCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Prelu_channel_shared", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Prelu_channel_slope", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(sigmoid,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::SigmoidCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    tanh, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::TanhCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    swish, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::SwishCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("beta", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    relu6, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::ReluCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
