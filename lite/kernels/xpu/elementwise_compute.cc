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

#include "lite/kernels/xpu/elementwise_compute.h"
#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
struct AddFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        const T* y,
                        T* z,
                        const std::vector<int>& xshape,
                        const std::vector<int>& yshape) const {
    return xdnn::broadcast_add<T>(ctx, x, y, z, xshape, yshape);
  }
};

template <typename T>
struct SubFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        const T* y,
                        T* z,
                        const std::vector<int>& xshape,
                        const std::vector<int>& yshape) const {
    return xdnn::broadcast_sub<T>(ctx, x, y, z, xshape, yshape);
  }
};

template <typename T>
struct MulFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        const T* y,
                        T* z,
                        const std::vector<int>& xshape,
                        const std::vector<int>& yshape) const {
    return xdnn::broadcast_mul<T>(ctx, x, y, z, xshape, yshape);
  }
};

template <typename T>
struct DivFunctor {
  inline int operator()(xdnn::Context* ctx,
                        const T* x,
                        const T* y,
                        T* z,
                        const std::vector<int>& xshape,
                        const std::vector<int>& yshape) const {
    return xdnn::broadcast_div<T>(ctx, x, y, z, xshape, yshape);
  }
};

template <class T, class Functor>
void ElementwiseCompute<T, Functor>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  const Tensor* x = param.X;
  const Tensor* y = param.Y;
  if (x->dims().size() < y->dims().size()) {
    std::swap(x, y);
  }

  auto& x_dim = x->dims();
  auto& y_dim = y->dims();
  CHECK_LE(y_dim.size(), x_dim.size());

  std::vector<int> x_shape(param.Out->dims().size(), 1);
  std::vector<int> y_shape(param.Out->dims().size(), 1);
  const int axis =
      (param.axis == -1 ? static_cast<int>(x_dim.size() - y_dim.size())
                        : param.axis);
  for (size_t i = 0; i < x_dim.size(); i++) {
    x_shape[i] = static_cast<int>(x_dim[i]);
  }
  for (size_t i = 0; i < y_dim.size(); ++i) {
    y_shape[i + axis] = static_cast<int>(y_dim[i]);
  }

  Functor elt_func;
  int ret = elt_func(ctx.GetRawContext(),
                     x->template data<T>(),
                     y->template data<T>(),
                     param.Out->template mutable_data<T>(TARGET(kXPU)),
                     x_shape,
                     y_shape);

  CHECK_EQ(ret, 0);
  return;
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;
using AddFloat32 = xpu::ElementwiseCompute<float, xpu::AddFunctor<float>>;
using AddInt32 = xpu::ElementwiseCompute<int, xpu::AddFunctor<int>>;
using SubFloat32 = xpu::ElementwiseCompute<float, xpu::SubFunctor<float>>;
using MulFloat32 = xpu::ElementwiseCompute<float, xpu::MulFunctor<float>>;
using DivFloat32 = xpu::ElementwiseCompute<float, xpu::DivFunctor<float>>;

REGISTER_LITE_KERNEL(elementwise_add, kXPU, kFloat, kNCHW, AddFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_add, kXPU, kFloat, kNCHW, AddInt32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub, kXPU, kFloat, kNCHW, SubFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul, kXPU, kFloat, kNCHW, MulFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div, kXPU, kFloat, kNCHW, DivFloat32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
