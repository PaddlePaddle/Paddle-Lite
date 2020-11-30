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

#include "lite/kernels/xpu/activation_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void ReluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::relu(ctx.GetRawContext(),
                     param.X->data<float>(),
                     param.Out->mutable_data<float>(TARGET(kXPU)),
                     param.X->numel());
  CHECK_EQ(r, 0);
}

void Relu6Compute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::relu6(ctx.GetRawContext(),
                      param.X->data<float>(),
                      param.Out->mutable_data<float>(TARGET(kXPU)),
                      param.X->numel());
  CHECK_EQ(r, 0);
}

void TanhCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::tanh(ctx.GetRawContext(),
                     param.X->data<float>(),
                     param.Out->mutable_data<float>(TARGET(kXPU)),
                     param.X->numel());
  CHECK_EQ(r, 0);
}

void SigmoidCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::sigmoid(ctx.GetRawContext(),
                        param.X->data<float>(),
                        param.Out->mutable_data<float>(TARGET(kXPU)),
                        param.X->numel());
  CHECK_EQ(r, 0);
}

void AbsCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::abs(ctx.GetRawContext(),
                    param.X->data<float>(),
                    param.Out->mutable_data<float>(TARGET(kXPU)),
                    param.X->numel());
  CHECK_EQ(r, 0);
}

void ExpCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::exp(ctx.GetRawContext(),
                    param.X->data<float>(),
                    param.Out->mutable_data<float>(TARGET(kXPU)),
                    param.X->numel());
  CHECK_EQ(r, 0);
}

void SquareCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::square(ctx.GetRawContext(),
                       param.X->data<float>(),
                       param.Out->mutable_data<float>(TARGET(kXPU)),
                       param.X->numel());
  CHECK_EQ(r, 0);
}

void ReciprocalCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r =
      xdnn::activation_forward(ctx.GetRawContext(),
                               xdnn::Activation_t::RECIPROCAL,
                               param.X->numel(),
                               param.X->data<float>(),
                               param.Out->mutable_data<float>(TARGET(kXPU)));
  CHECK_EQ(r, 0);
}

void SqrtCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::sqrt(ctx.GetRawContext(),
                     param.X->data<float>(),
                     param.Out->mutable_data<float>(TARGET(kXPU)),
                     param.X->numel());
  CHECK_EQ(r, 0);
}

void PowCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  xdnn::Activation_t act_type(xdnn::Activation_t::ACT_POW);
  act_type.pow_factor = param.factor;

  int r =
      xdnn::activation_forward(ctx.GetRawContext(),
                               act_type,
                               param.X->numel(),
                               param.X->data<float>(),
                               param.Out->mutable_data<float>(TARGET(kXPU)));
  CHECK_EQ(r, 0);
}

void SignCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r =
      xdnn::activation_forward(ctx.GetRawContext(),
                               xdnn::Activation_t::SIGN,
                               param.X->numel(),
                               param.X->data<float>(),
                               param.Out->mutable_data<float>(TARGET(kXPU)));
  CHECK_EQ(r, 0);
}

void HardSwishCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::hard_swish(ctx.GetRawContext(),
                           param.X->data<float>(),
                           param.Out->mutable_data<float>(TARGET(kXPU)),
                           param.X->numel());
  CHECK_EQ(r, 0);
}

void HardSigmoidCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::hard_sigmoid(ctx.GetRawContext(),
                             param.X->data<float>(),
                             param.Out->mutable_data<float>(TARGET(kXPU)),
                             param.X->numel(),
                             param.hard_sigmoid_slope);
  CHECK_EQ(r, 0);
}

void LeakyReluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::leaky_relu(ctx.GetRawContext(),
                           param.X->data<float>(),
                           param.Out->mutable_data<float>(TARGET(kXPU)),
                           param.X->numel(),
                           param.Leaky_relu_alpha);
  CHECK_EQ(r, 0);
}

void SoftsignCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::softsign(ctx.GetRawContext(),
                         param.X->data<float>(),
                         param.Out->mutable_data<float>(TARGET(kXPU)),
                         param.X->numel());
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    relu, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::ReluCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    relu6, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::Relu6Compute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    tanh, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::TanhCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(sigmoid,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SigmoidCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    abs, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::AbsCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    exp, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::ExpCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    square, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::SquareCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    sqrt, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::SqrtCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    pow, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::PowCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    sign, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::SignCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(reciprocal,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ReciprocalCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(hard_sigmoid,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::HardSigmoidCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(hard_swish,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::HardSwishCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(leaky_relu,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::LeakyReluCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(softsign,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SoftsignCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
