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

template <typename T, PrecisionType PType>
void ReluCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::relu(ctx.GetRawContext(),
                     param.X->template data<T>(),
                     param.Out->template mutable_data<T>(TARGET(kXPU)),
                     param.X->numel());
  CHECK_EQ(r, 0);
}

template <typename T, PrecisionType PType>
void Relu6Compute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::relu6(ctx.GetRawContext(),
                      param.X->template data<T>(),
                      param.Out->template mutable_data<T>(TARGET(kXPU)),
                      param.X->numel());
  CHECK_EQ(r, 0);
}

template <typename T, PrecisionType PType>
void GeluCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::gelu(ctx.GetRawContext(),
                     param.X->template data<T>(),
                     param.Out->template mutable_data<T>(TARGET(kXPU)),
                     param.X->numel());
  CHECK_EQ(r, 0);
}

template <typename T, PrecisionType PType>
void TanhCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::fast_tanh(ctx.GetRawContext(),
                          param.X->template data<T>(),
                          param.Out->template mutable_data<T>(TARGET(kXPU)),
                          param.X->numel());
  CHECK_EQ(r, 0);
}

template <typename T, PrecisionType PType>
void SigmoidCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::fast_sigmoid(ctx.GetRawContext(),
                             param.X->template data<T>(),
                             param.Out->template mutable_data<T>(TARGET(kXPU)),
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

  XPUScratchPadGuard xpu_factor_guard =
      TargetWrapperXPU::MallocScratchPad(sizeof(float));
  float* xpu_factor_ptr = reinterpret_cast<float*>(xpu_factor_guard->addr_);
  int x_len = param.X->numel();
  int r = 0;
  r = xdnn::constant<float>(ctx.GetRawContext(), xpu_factor_ptr, 1, 1.0f);
  CHECK_EQ(r, 0);
  r = xdnn::broadcast_div(ctx.GetRawContext(),
                          xpu_factor_ptr,
                          param.X->data<float>(),
                          param.Out->mutable_data<float>(TARGET(kXPU)),
                          {1},
                          {x_len});
  CHECK_EQ(r, 0);
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

  XPUScratchPadGuard xpu_factor_guard =
      TargetWrapperXPU::MallocScratchPad(sizeof(float));
  float* xpu_factor_ptr = reinterpret_cast<float*>(xpu_factor_guard->addr_);
  int x_len = param.X->numel();
  int r = 0;
  r = xdnn::constant<float>(
      ctx.GetRawContext(), xpu_factor_ptr, 1, param.factor);
  CHECK_EQ(r, 0);
  r = xdnn::broadcast_pow(ctx.GetRawContext(),
                          param.X->data<float>(),
                          xpu_factor_ptr,
                          param.Out->mutable_data<float>(TARGET(kXPU)),
                          {x_len},
                          {1});
  CHECK_EQ(r, 0);
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

template <typename T, PrecisionType PType>
void LeakyReluCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  int r = xdnn::leaky_relu(ctx.GetRawContext(),
                           param.X->template data<T>(),
                           param.Out->template mutable_data<T>(TARGET(kXPU)),
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

void FloorCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int r = xdnn::floor(ctx.GetRawContext(),
                      param.X->data<float>(),
                      param.Out->mutable_data<float>(TARGET(kXPU)),
                      param.X->numel());
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using reluFP32 =
    paddle::lite::kernels::xpu::ReluCompute<float, PRECISION(kFloat)>;
using reluFP16 =
    paddle::lite::kernels::xpu::ReluCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(relu, kXPU, kFloat, kNCHW, reluFP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(relu, kXPU, kFP16, kNCHW, reluFP16, reluFP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

using relu6FP32 =
    paddle::lite::kernels::xpu::Relu6Compute<float, PRECISION(kFloat)>;
using relu6FP16 =
    paddle::lite::kernels::xpu::Relu6Compute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(relu6, kXPU, kFloat, kNCHW, relu6FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(relu6, kXPU, kFP16, kNCHW, relu6FP16, relu6FP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

using geluFP32 =
    paddle::lite::kernels::xpu::GeluCompute<float, PRECISION(kFloat)>;
using geluFP16 =
    paddle::lite::kernels::xpu::GeluCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(gelu, kXPU, kFloat, kNCHW, geluFP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
using gelu_fp16 =
    paddle::lite::kernels::xpu::GeluCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(gelu, kXPU, kFP16, kNCHW, geluFP16, geluFP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

using tanhFP32 =
    paddle::lite::kernels::xpu::TanhCompute<float, PRECISION(kFloat)>;
using tanhFP16 =
    paddle::lite::kernels::xpu::TanhCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(tanh, kXPU, kFloat, kNCHW, tanhFP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(tanh, kXPU, kFP16, kNCHW, tanhFP16, tanhFP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

using sigmoidFP32 =
    paddle::lite::kernels::xpu::SigmoidCompute<float, PRECISION(kFloat)>;
using sigmoidFP16 =
    paddle::lite::kernels::xpu::SigmoidCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(sigmoid, kXPU, kFloat, kNCHW, sigmoidFP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(sigmoid, kXPU, kFP16, kNCHW, sigmoidFP16, sigmoidFP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
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

using leaky_reluFP32 =
    paddle::lite::kernels::xpu::LeakyReluCompute<float, PRECISION(kFloat)>;
using leaky_reluFP16 =
    paddle::lite::kernels::xpu::LeakyReluCompute<float16, PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(leaky_relu, kXPU, kFloat, kNCHW, leaky_reluFP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    leaky_relu, kXPU, kFP16, kNCHW, leaky_reluFP16, leaky_reluFP16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
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

REGISTER_LITE_KERNEL(
    floor, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::FloorCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
