// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <functional>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

struct LogicalAndFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const bool* x,
                        const bool* y,
                        bool* z,
                        int len) const {
    return xdnn::logical_and<bool>(ctx, x, y, z, len);
  }
};

struct LogicalNotFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const bool* x,
                        bool* z,
                        int len) const {
    return xdnn::logical_not<bool>(ctx, x, z, len);
  }
};

template <class Functor>
void BinaryLogicalCompute<Functor>::Run() {
  auto& param = this->template Param<operators::LogicalParam>();
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>(TARGET(kXPU));
  const auto* x = param.X->template data<bool>();
  const auto* y = param.Y->template data<bool>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  Functor binary_logic_func;
  int r = binary_logic_func(ctx.GetRawContext(), x, y, z, count);
  CHECK_EQ(r, 0);
}

template <class Functor>
void UnaryLogicalCompute<Functor>::Run() {
  auto& param = this->template Param<operators::LogicalParam>();
  const size_t count = param.X->numel();
  bool* z = param.Out->template mutable_data<bool>(TARGET(kXPU));
  const auto* x = param.X->template data<bool>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  Functor unary_logic_func;
  int r = unary_logic_func(ctx.GetRawContext(), x, z, count);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using LogicalAnd = paddle::lite::kernels::xpu::BinaryLogicalCompute<
    paddle::lite::kernels::xpu::LogicalAndFunctor>;

REGISTER_LITE_KERNEL(logical_and, kXPU, kFloat, kAny, LogicalAnd, def)
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

using LogicalNot = paddle::lite::kernels::xpu::UnaryLogicalCompute<
    paddle::lite::kernels::xpu::LogicalNotFunctor>;

REGISTER_LITE_KERNEL(logical_not, kXPU, kFloat, kAny, LogicalNot, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .Finalize();
