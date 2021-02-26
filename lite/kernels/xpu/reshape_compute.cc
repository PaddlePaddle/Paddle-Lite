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

#include "lite/kernels/xpu/reshape_compute.h"
#include <algorithm>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void ReshapeCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x = param.x;
  auto output = param.output;
  auto output_dims = output->dims();

  if (param.inplace) {
    output->ShareDataWith(*x);
    output->Resize(output_dims);
  } else {
    int r = xdnn::copy<T>(ctx.GetRawContext(),
                          x->template data<T>(),
                          output->template mutable_data<T>(TARGET(kXPU)),
                          x->numel());

    CHECK_EQ(r, 0);
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reshape2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ReshapeCompute<float>,
                     float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ReshapeCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ReshapeCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ReshapeCompute<float>,
                     float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ReshapeCompute<float>,
                     float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::ReshapeCompute<float>,
                     float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
