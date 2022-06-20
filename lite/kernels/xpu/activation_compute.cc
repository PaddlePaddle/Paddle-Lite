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
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::relu(ctx.GetRawContext(),
                     param.X->data<float>(),
                     param.Out->mutable_data<float>(TARGET(kXPU)),
                     param.X->numel());
  CHECK_EQ(r, 0);
}

void Relu6Compute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::relu6(ctx.GetRawContext(),
                      param.X->data<float>(),
                      param.Out->mutable_data<float>(TARGET(kXPU)),
                      param.X->numel());
  CHECK_EQ(r, 0);
}

void GeluCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::gelu(ctx.GetRawContext(),
                     param.X->data<float>(),
                     param.Out->mutable_data<float>(TARGET(kXPU)),
                     param.X->numel());
  CHECK_EQ(r, 0);
}

void TanhCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::tanh(ctx.GetRawContext(),
                     param.X->data<float>(),
                     param.Out->mutable_data<float>(TARGET(kXPU)),
                     param.X->numel());
  CHECK_EQ(r, 0);
}

void SigmoidCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::sigmoid(ctx.GetRawContext(),
                        param.X->data<float>(),
                        param.Out->mutable_data<float>(TARGET(kXPU)),
                        param.X->numel());
  CHECK_EQ(r, 0);
}

void AbsCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::abs(ctx.GetRawContext(),
                    param.X->data<float>(),
                    param.Out->mutable_data<float>(TARGET(kXPU)),
                    param.X->numel());
  CHECK_EQ(r, 0);
}

void ExpCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::exp(ctx.GetRawContext(),
                    param.X->data<float>(),
                    param.Out->mutable_data<float>(TARGET(kXPU)),
                    param.X->numel());
  CHECK_EQ(r, 0);
}

void SquareCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::square(ctx.GetRawContext(),
                       param.X->data<float>(),
                       param.Out->mutable_data<float>(TARGET(kXPU)),
                       param.X->numel());
  CHECK_EQ(r, 0);
}

void ReciprocalCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  float* xpu_factor = nullptr;
  XPU_CALL(xpu_malloc(reinterpret_cast<void**>(&xpu_factor), sizeof(float)));
  int x_len = param.X->numel();
  int r = 0;
  r = xdnn::constant<float>(ctx.GetRawContext(), xpu_factor, 1, 1.0f);
  CHECK_EQ(r, 0);
  r = xdnn::broadcast_div(ctx.GetRawContext(),
                          xpu_factor,
                          param.X->data<float>(),
                          param.Out->mutable_data<float>(TARGET(kXPU)),
                          {1},
                          {x_len});
  CHECK_EQ(r, 0);
  XPU_CALL(xpu_free(xpu_factor));
}

void SqrtCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::sqrt(ctx.GetRawContext(),
                     param.X->data<float>(),
                     param.Out->mutable_data<float>(TARGET(kXPU)),
                     param.X->numel());
  CHECK_EQ(r, 0);
}

void RsqrtCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::rsqrt(ctx.GetRawContext(),
                      param.X->data<float>(),
                      param.Out->mutable_data<float>(TARGET(kXPU)),
                      param.X->numel());
  CHECK_EQ(r, 0);
}

void PowCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  float* xpu_factor = nullptr;
  XPU_CALL(xpu_malloc(reinterpret_cast<void**>(&xpu_factor), sizeof(float)));
  int x_len = param.X->numel();
  int r = 0;
  r = xdnn::constant<float>(ctx.GetRawContext(), xpu_factor, 1, param.factor);
  CHECK_EQ(r, 0);
  r = xdnn::broadcast_pow(ctx.GetRawContext(),
                          param.X->data<float>(),
                          xpu_factor,
                          param.Out->mutable_data<float>(TARGET(kXPU)),
                          {x_len},
                          {1});
  CHECK_EQ(r, 0);
  XPU_CALL(xpu_free(xpu_factor));
}

void SignCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::sign(ctx.GetRawContext(),
                     param.X->data<float>(),
                     param.Out->mutable_data<float>(TARGET(kXPU)),
                     param.X->numel());
  CHECK_EQ(r, 0);
}

void HardSwishCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::hard_swish(ctx.GetRawContext(),
                           param.X->data<float>(),
                           param.Out->mutable_data<float>(TARGET(kXPU)),
                           param.X->numel());
  CHECK_EQ(r, 0);
}

void HardSigmoidCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::hard_sigmoid(ctx.GetRawContext(),
                             param.X->data<float>(),
                             param.Out->mutable_data<float>(TARGET(kXPU)),
                             param.X->numel(),
                             param.hard_sigmoid_slope);
  CHECK_EQ(r, 0);
}

void LeakyReluCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::leaky_relu(ctx.GetRawContext(),
                           param.X->data<float>(),
                           param.Out->mutable_data<float>(TARGET(kXPU)),
                           param.X->numel(),
                           param.Leaky_relu_alpha);
  CHECK_EQ(r, 0);
}

void LogCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::log<float>(ctx.GetRawContext(),    /* context */
                           param.X->data<float>(), /* x */
                           param.Out->mutable_data<float>(TARGET(kXPU)), /* y */
                           param.X->numel()); /* len */
  CHECK_EQ(r, 0);
}

void SoftsignCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::softsign(ctx.GetRawContext(),
                         param.X->data<float>(),
                         param.Out->mutable_data<float>(TARGET(kXPU)),
                         param.X->numel());
  CHECK_EQ(r, 0);
}

void SwishCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto beta = param.Swish_beta;
  CHECK(std::abs(beta - 1.0f) < 1e-7);
  int r = xdnn::swish(ctx.GetRawContext(),
                      param.X->data<float>(),
                      param.Out->mutable_data<float>(TARGET(kXPU)),
                      param.X->numel());
  CHECK_EQ(r, 0);
}

void PReluCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x_dims = param.X->dims();
  int outer_size = x_dims[0];
  int channel_size = param.Prelu_alpha->numel();
  int inner_size = x_dims.count(1, x_dims.size()) / channel_size;

  int r = xdnn::prelu(ctx.GetRawContext(),
                      param.X->data<float>(),
                      param.Prelu_alpha->data<float>(),
                      param.Out->mutable_data<float>(TARGET(kXPU)),
                      outer_size,
                      channel_size,
                      inner_size);
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
    gelu, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::GeluCompute, def)
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
    rsqrt, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::RsqrtCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    pow, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::PowCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    log, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::LogCompute, def)
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

REGISTER_LITE_KERNEL(
    swish, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::SwishCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("beta", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    prelu, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::PReluCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Alpha", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
