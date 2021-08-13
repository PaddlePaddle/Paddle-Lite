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

#include "lite/kernels/xpu/compare_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
struct LessThanFunctor {
  inline int operator()(
      xdnn::Context* ctx, const T* x, const T* y, bool* z, int len) const {
    return xdnn::less_than<T>(ctx, x, y, z, len);
  }
};

template <PrecisionType PType, class T, class Functor>
void CompareCompute<PType, T, Functor>::Run() {
  auto& param = this->template Param<operators::CompareParam>();
  const size_t x_size = param.X->numel();
  const size_t y_size = param.Y->numel();
  bool* z = param.Out->template mutable_data<bool>(TARGET(kXPU));
  const auto* x = param.X->template data<T>();
  const auto* y = param.Y->template data<T>();

  auto& ctx = this->ctx_->template As<XPUContext>();
  if (x_size == y_size) {
    Functor comp_func;
    int r = comp_func(ctx.GetRawContext(), x, y, z, x_size);
    CHECK_EQ(r, 0);
  } else {
    LOG(FATAL) << "CompareCompute only supports x_size == y_size for "
                  "xpu at this moment, however them are not equal now";
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using less_than_float = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    float,
    paddle::lite::kernels::xpu::LessThanFunctor<float>>;
REGISTER_LITE_KERNEL(less_than, kXPU, kFloat, kAny, less_than_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kXPU), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .BindPaddleOpVersion("less_than", 1)
    .Finalize();

using less_than_int32 = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kInt32),
    int,
    paddle::lite::kernels::xpu::LessThanFunctor<int>>;
REGISTER_LITE_KERNEL(less_than, kXPU, kInt32, kAny, less_than_int32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kXPU), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .BindPaddleOpVersion("less_than", 1)
    .Finalize();

using less_than_int64 = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kInt64),
    int64_t,
    paddle::lite::kernels::xpu::LessThanFunctor<int64_t>>;
REGISTER_LITE_KERNEL(less_than, kXPU, kInt64, kAny, less_than_int64, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kInt64), DATALAYOUT(kAny), -1)})
    .BindInput("Y",
               {LiteType::GetTensorTy(
                   TARGET(kXPU), PRECISION(kInt64), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kXPU), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .BindPaddleOpVersion("less_than", 1)
    .Finalize();
