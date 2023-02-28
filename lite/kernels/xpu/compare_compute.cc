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
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
struct LessThanFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        const T* y,
                        bool* z,
                        const std::vector<int>& xshape,
                        const std::vector<int>& yshape) const {
    return xdnn::broadcast_less_than<T>(ctx, x, y, z, xshape, yshape);
  }
};

template <typename T>
struct EqualFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        const T* y,
                        bool* z,
                        const std::vector<int>& xshape,
                        const std::vector<int>& yshape) const {
    return xdnn::broadcast_equal<T>(ctx, x, y, z, xshape, yshape);
  }
};

template <typename T>
struct GreaterThanFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        const T* y,
                        bool* z,
                        const std::vector<int>& xshape,
                        const std::vector<int>& yshape) const {
    return xdnn::broadcast_greater_than<T>(ctx, x, y, z, xshape, yshape);
  }
};

template <typename T>
struct GreaterEqualFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        const T* y,
                        bool* z,
                        const std::vector<int>& xshape,
                        const std::vector<int>& yshape) const {
    return xdnn::broadcast_greater_equal<T>(ctx, x, y, z, xshape, yshape);
  }
};

template <PrecisionType PType, class T, class Functor>
void CompareCompute<PType, T, Functor>::Run() {
  auto& param = this->template Param<operators::CompareParam>();
  const size_t x_size = param.X->numel();
  const size_t y_size = param.Y->numel();
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  bool* z = param.Out->template mutable_data<bool>(TARGET(kXPU));
  const auto* x = param.X->template data<T>();
  const auto* y = param.Y->template data<T>();

  auto& ctx = this->ctx_->template As<XPUContext>();
  Functor comp_func;
  std::vector<int> xshape;
  std::vector<int> yshape;
  int axis = (param.axis == -1 ? abs(static_cast<int>(x_dims.size()) -
                                     static_cast<int>(y_dims.size()))
                               : param.axis);
  // constrains:
  // 1. X size should be larger than Y
  CHECK_GE(x_size, y_size) << "Input X cannot be smaller than Y";
  CHECK_GE(x_dims.size(), y_dims.size())
      << "Axis number of X cannot be less than Y";

  // 2. y_dims.size() + axis <= x_dims.size()
  CHECK_LE(y_dims.size() + axis, x_dims.size()) << "Invalid param.axis";

  for (int i = 0; i < x_dims.size(); ++i) {
    CHECK_LE(x_dims[i], INT_MAX) << "Dimension of X exceed INT_MAX";
    xshape.push_back(static_cast<int>(x_dims[i]));
  }

  // yshape should be:
  // [0, axis)                       = 1
  // [axis, axis + y_dims.size)      = y_dims
  // [axis+y_dims.size, x_dims.size) = 1
  for (int i = 0; i < axis; ++i) {
    yshape.push_back(1);
  }

  for (int i = 0; i < y_dims.size(); ++i) {
    CHECK_LE(y_dims[i], INT_MAX) << "Dimension of Y exceed INT_MAX";
    yshape.push_back(static_cast<int>(y_dims[i]));
  }

  for (int i = 0; i < x_dims.size() - y_dims.size() - axis; ++i) {
    yshape.push_back(1);
  }

  int r = comp_func(ctx.GetRawContext(), x, y, z, xshape, yshape);
  CHECK_EQ(r, 0);
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
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_than", 1)
    .Finalize();

using less_than_int32 = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    int,
    paddle::lite::kernels::xpu::LessThanFunctor<int>>;
REGISTER_LITE_KERNEL(less_than, kXPU, kFloat, kAny, less_than_int32, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_than", 1)
    .Finalize();

using less_than_int64 = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    int64_t,
    paddle::lite::kernels::xpu::LessThanFunctor<int64_t>>;
REGISTER_LITE_KERNEL(less_than, kXPU, kFloat, kAny, less_than_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("less_than", 1)
    .Finalize();

using equal_float = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    float,
    paddle::lite::kernels::xpu::EqualFunctor<float>>;
REGISTER_LITE_KERNEL(equal, kXPU, kFloat, kAny, equal_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("equal", 1)
    .Finalize();

using equal_int32 = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    int,
    paddle::lite::kernels::xpu::EqualFunctor<int>>;
REGISTER_LITE_KERNEL(equal, kXPU, kFloat, kAny, equal_int32, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("equal", 1)
    .Finalize();

using euqal_int64 = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    int64_t,
    paddle::lite::kernels::xpu::EqualFunctor<int64_t>>;
REGISTER_LITE_KERNEL(equal, kXPU, kFloat, kAny, euqal_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("equal", 1)
    .Finalize();

using greater_than_float = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    float,
    paddle::lite::kernels::xpu::GreaterThanFunctor<float>>;
REGISTER_LITE_KERNEL(greater_than, kXPU, kFloat, kAny, greater_than_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_than", 1)
    .Finalize();

using greater_than_int32 = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    int,
    paddle::lite::kernels::xpu::GreaterThanFunctor<int>>;
REGISTER_LITE_KERNEL(
    greater_than, kXPU, kFloat, kAny, greater_than_int32, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_than", 1)
    .Finalize();

using greater_than_int64 = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    int64_t,
    paddle::lite::kernels::xpu::GreaterThanFunctor<int64_t>>;
REGISTER_LITE_KERNEL(
    greater_than, kXPU, kFloat, kAny, greater_than_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_than", 1)
    .Finalize();

using greater_equal_float = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    float,
    paddle::lite::kernels::xpu::GreaterEqualFunctor<float>>;
REGISTER_LITE_KERNEL(
    greater_equal, kXPU, kFloat, kAny, greater_equal_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_equal", 1)
    .Finalize();

using greater_equal_int32 = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    int,
    paddle::lite::kernels::xpu::GreaterEqualFunctor<int>>;
REGISTER_LITE_KERNEL(
    greater_equal, kXPU, kFloat, kAny, greater_equal_int32, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_equal", 1)
    .Finalize();

using greater_equal_int64 = paddle::lite::kernels::xpu::CompareCompute<
    PRECISION(kFloat),
    int64_t,
    paddle::lite::kernels::xpu::GreaterEqualFunctor<int64_t>>;
REGISTER_LITE_KERNEL(
    greater_equal, kXPU, kFloat, kAny, greater_equal_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kBool),
                                       DATALAYOUT(kAny))})
    .BindPaddleOpVersion("greater_equal", 1)
    .Finalize();
