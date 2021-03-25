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

#include "lite/kernels/xpu/logical_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <>
void BinaryLogicalCompute<1>::Run() {
  auto& param = this->Param<operators::LogicalParam>();
  auto& ctx = this->ctx_->As<XPUContext>();
  CHECK(sizeof(bool) == 1) << " unsupported bool size: " << sizeof(bool);
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>(TARGET(kXPU));
  const bool* x = param.X->template data<bool>();
  const bool* y = param.Y->template data<bool>();
  int r = xdnn::logical_and<bool>(ctx.GetRawContext(), x, y, z, count);
  CHECK(r == 0) << " xpu logical_and failed";
}

template <>
void BinaryLogicalCompute<2>::Run() {
  auto& param = this->Param<operators::LogicalParam>();
  auto& ctx = this->ctx_->As<XPUContext>();
  CHECK(sizeof(bool) == 1) << " unsupported bool size: " << sizeof(bool);
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>(TARGET(kXPU));
  const bool* x = param.X->template data<bool>();
  const bool* y = param.Y->template data<bool>();
  int r = xdnn::logical_or<bool>(ctx.GetRawContext(), x, y, z, count);
  CHECK(r == 0) << " xpu logical_or failed";
}

template <>
void BinaryLogicalCompute<3>::Run() {
  auto& param = this->Param<operators::LogicalParam>();
  auto& ctx = this->ctx_->As<XPUContext>();
  CHECK(sizeof(bool) == 1) << " unsupported bool size: " << sizeof(bool);
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>(TARGET(kXPU));
  const bool* x = param.X->template data<bool>();
  const bool* y = param.Y->template data<bool>();
  int r = xdnn::logical_xor<bool>(ctx.GetRawContext(), x, y, z, count);
  CHECK(r == 0) << " xpu logical_xor failed";
}

void UnaryLogicalCompute::Run() {
  auto& param = this->Param<operators::LogicalParam>();
  auto& ctx = this->ctx_->As<XPUContext>();
  CHECK(sizeof(bool) == 1) << " unsupported bool size: " << sizeof(bool);
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>(TARGET(kXPU));
  const auto x = param.X->template data<bool>();
  int r = xdnn::logical_not<bool>(ctx.GetRawContext(), x, z, count);
  CHECK(r == 0) << " xpu logical_not failed";
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(logical_and,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::BinaryLogicalCompute<1>,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(logical_or,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::BinaryLogicalCompute<2>,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(logical_xor,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::BinaryLogicalCompute<3>,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(logical_not,
                     kXPU,
                     kAny,
                     kAny,
                     paddle::lite::kernels::xpu::UnaryLogicalCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();
