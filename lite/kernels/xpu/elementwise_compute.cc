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
#include <functional>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void ElementwiseAddCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& x_dims = param.X->dims().data();
  auto& y_dims = param.Y->dims();
  int axis = param.axis;
  if (param.axis == -1) {
    axis = x_dims.size() - y_dims.size();
  }
  int iter = std::accumulate(
      x_dims.begin(), x_dims.begin() + axis, 1, std::multiplies<int>());
  int stride = param.Y->numel();

  for (int i = 0; i < iter; ++i) {
    const float* x_ptr = param.X->data<float>() + i * stride;
    const float* y_ptr = param.Y->data<float>();
    float* o_ptr = param.Out->mutable_data<float>(TARGET(kXPU)) + i * stride;
    int r = xdnn::elementwise_add(ctx.GetRawContext(), /* context */
                                  x_ptr,               /* x */
                                  y_ptr,               /* y */
                                  o_ptr,               /* z */
                                  stride /* len */);
    CHECK_EQ(r, 0);
  }
}

void ElementwiseSubCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto& x_dims = param.X->dims().data();
  auto& y_dims = param.Y->dims();
  int axis = param.axis;
  if (param.axis == -1) {
    axis = x_dims.size() - y_dims.size();
  }
  int iter = std::accumulate(
      x_dims.begin(), x_dims.begin() + axis, 1, std::multiplies<int>());
  int stride = param.Y->numel();

  for (int i = 0; i < iter; ++i) {
    const float* x_ptr = param.X->data<float>() + i * stride;
    const float* y_ptr = param.Y->data<float>();
    float* o_ptr = param.Out->mutable_data<float>(TARGET(kXPU)) + i * stride;
    int r = xdnn::elementwise_sub(ctx.GetRawContext(), /* context */
                                  x_ptr,               /* x */
                                  y_ptr,               /* y */
                                  o_ptr,               /* z */
                                  stride /* len */);
    CHECK_EQ(r, 0);
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(elementwise_add,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ElementwiseAddCompute,
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
