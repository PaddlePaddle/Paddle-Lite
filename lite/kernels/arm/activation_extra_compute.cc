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

#include "lite/kernels/arm/activation_extra_compute.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

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

void LogCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_log<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void ExpCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_exp<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void FloorCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_floor<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void HardSigmoidCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  float slope = param.hard_sigmoid_slope;
  float offset = param.hard_sigmoid_offset;
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_hard_sigmoid<float>(
      x_data, output_data, x_dims.production(), slope, offset, ctx.threads());
}

void SqrtCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_sqrt<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void RsqrtCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_rsqrt<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void SquareCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_square<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void HardSwishCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  float threshold = param.hard_swish_threshold;
  float scale = param.hard_swish_scale;
  float offset = param.hard_swish_offset;
  lite::arm::math::act_hard_swish<float>(x_data,
                                         output_data,
                                         x_dims.production(),
                                         threshold,
                                         scale,
                                         offset,
                                         ctx.threads());
}

void ReciprocalCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_reciprocal<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void AbsCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  lite::arm::math::act_abs<float>(
      x_data, output_data, x_dims.production(), ctx.threads());
}

void GeluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto x_dims = param.X->dims();
  auto x_data = param.X->data<float>();
  auto output_data = param.Out->mutable_data<float>();
  bool approximate = param.gelu_approximate;
  lite::arm::math::act_gelu<float>(
      x_data, output_data, x_dims.production(), approximate, ctx.threads());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

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
    swish, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::SwishCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("beta", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    log, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::LogCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    exp, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::ExpCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    floor, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::FloorCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(hard_sigmoid,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::HardSigmoidCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    sqrt, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::SqrtCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    rsqrt, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::RsqrtCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    square, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::SquareCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(hard_swish,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::HardSwishCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(reciprocal,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ReciprocalCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
REGISTER_LITE_KERNEL(
    abs, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::AbsCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    gelu, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::GeluCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
