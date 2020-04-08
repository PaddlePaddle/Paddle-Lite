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

  int r = xdnn::activation_forward(
      ctx.GetRawContext(),      /* context */
      xdnn::Activation_t::RELU, /* type */
      param.X->numel(),         /* len */
      param.X->data<float>(),   /* x */
      param.Out->mutable_data<float>(TARGET(kXPU)) /* y */);
  CHECK_EQ(r, 0);
}

void TanhCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::activation_forward(
      ctx.GetRawContext(),      /* context */
      xdnn::Activation_t::TANH, /* type */
      param.X->numel(),         /* len */
      param.X->data<float>(),   /* x */
      param.Out->mutable_data<float>(TARGET(kXPU)) /* y */);
  CHECK_EQ(r, 0);
}

void SigmoidCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  int r = xdnn::activation_forward(
      ctx.GetRawContext(),         /* context */
      xdnn::Activation_t::SIGMOID, /* type */
      param.X->numel(),            /* len */
      param.X->data<float>(),      /* x */
      param.Out->mutable_data<float>(TARGET(kXPU)) /* y */);
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
