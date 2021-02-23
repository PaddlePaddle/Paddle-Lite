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
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void ElementwiseAddCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto& x_dim = param.X->dims();
  auto& y_dim = param.Y->dims();

  std::vector<int> x_shape(param.Out->dims().size(), 1);
  std::vector<int> y_shape(param.Out->dims().size(), 1);
  int axis = (param.axis == -1
                  ? std::abs(static_cast<int>(x_dim.size() - y_dim.size()))
                  : param.axis);
  CHECK_LE(y_dim.size(), x_dim.size());
  for (size_t i = 0; i < x_dim.size(); i++) {
    x_shape[i] = static_cast<int>(x_dim[i]);
  }
  for (size_t i = 0; i < y_dim.size(); ++i) {
    y_shape[i + axis] = static_cast<int>(y_dim[i]);
  }

  int ret =
      xdnn::broadcast_add<T>(ctx.GetRawContext(),
                             param.X->template data<T>(),
                             param.Y->template data<T>(),
                             param.Out->template mutable_data<T>(TARGET(kXPU)),
                             x_shape,
                             y_shape);

  CHECK_EQ(ret, 0);
  return;
}

void ElementwiseMulCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& x_dim = param.X->dims();
  auto& y_dim = param.Y->dims();

  std::vector<int> x_shape(param.Out->dims().size(), 1);
  std::vector<int> y_shape(param.Out->dims().size(), 1);
  int axis = (param.axis == -1
                  ? std::abs(static_cast<int>(x_dim.size() - y_dim.size()))
                  : param.axis);
  CHECK_LE(y_dim.size(), x_dim.size());
  for (size_t i = 0; i < x_dim.size(); i++) {
    x_shape[i] = static_cast<int>(x_dim[i]);
  }
  for (size_t i = 0; i < y_dim.size(); ++i) {
    y_shape[i + axis] = static_cast<int>(y_dim[i]);
  }

  int ret =
      xdnn::broadcast_mul<float>(ctx.GetRawContext(),
                                 param.X->data<float>(),
                                 param.Y->data<float>(),
                                 param.Out->mutable_data<float>(TARGET(kXPU)),
                                 x_shape,
                                 y_shape);

  CHECK_EQ(ret, 0);
  return;
}

void ElementwiseSubCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& x_dim = param.X->dims();
  auto& y_dim = param.Y->dims();

  std::vector<int> x_shape(param.Out->dims().size(), 1);
  std::vector<int> y_shape(param.Out->dims().size(), 1);
  int axis = (param.axis == -1
                  ? std::abs(static_cast<int>(x_dim.size() - y_dim.size()))
                  : param.axis);
  CHECK_LE(y_dim.size(), x_dim.size());
  for (size_t i = 0; i < x_dim.size(); i++) {
    x_shape[i] = static_cast<int>(x_dim[i]);
  }
  for (size_t i = 0; i < y_dim.size(); ++i) {
    y_shape[i + axis] = static_cast<int>(y_dim[i]);
  }

  int ret =
      xdnn::broadcast_sub<float>(ctx.GetRawContext(),
                                 param.X->data<float>(),
                                 param.Y->data<float>(),
                                 param.Out->mutable_data<float>(TARGET(kXPU)),
                                 x_shape,
                                 y_shape);

  CHECK_EQ(ret, 0);
  return;
}

void ElementwiseDivCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& x_dim = param.X->dims();
  auto& y_dim = param.Y->dims();

  std::vector<int> x_shape(param.Out->dims().size(), 1);
  std::vector<int> y_shape(param.Out->dims().size(), 1);
  int axis = (param.axis == -1
                  ? std::abs(static_cast<int>(x_dim.size() - y_dim.size()))
                  : param.axis);
  CHECK_LE(y_dim.size(), x_dim.size());
  for (size_t i = 0; i < x_dim.size(); i++) {
    x_shape[i] = static_cast<int>(x_dim[i]);
  }
  for (size_t i = 0; i < y_dim.size(); ++i) {
    y_shape[i + axis] = static_cast<int>(y_dim[i]);
  }

  int ret =
      xdnn::broadcast_div<float>(ctx.GetRawContext(),
                                 param.X->data<float>(),
                                 param.Y->data<float>(),
                                 param.Out->mutable_data<float>(TARGET(kXPU)),
                                 x_shape,
                                 y_shape);

  CHECK_EQ(ret, 0);
  return;
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ElementwiseAddCompute<float>,
                     float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_add,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ElementwiseAddCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ElementwiseMulCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ElementwiseSubCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_div,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ElementwiseDivCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
