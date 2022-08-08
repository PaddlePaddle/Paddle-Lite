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

template <typename T, PrecisionType PType>
void ReshapeCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x = param.x;
  auto output = param.output;
  auto output_dims = output->dims();
  if (output_dims.production() == 0) {
    output->set_target(TARGET(kXPU));
    return;
  }

  if (param.inplace) {
    auto output_lod = output->lod();
    output->ShareDataWith(*x);
    output->Resize(output_dims);
    output->set_lod(output_lod);
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

using reshape2FP32 =
    paddle::lite::kernels::xpu::ReshapeCompute<float, PRECISION(kFloat)>;
using reshape2FP16 =
    paddle::lite::kernels::xpu::ReshapeCompute<float, PRECISION(kFP16)>;
using reshape2Int32 =
    paddle::lite::kernels::xpu::ReshapeCompute<int, PRECISION(kFloat)>;
using reshape2Int64 =
    paddle::lite::kernels::xpu::ReshapeCompute<int64_t, PRECISION(kFloat)>;
using reshape2INT8 =
    paddle::lite::kernels::xpu::ReshapeCompute<int8_t, PRECISION(kInt8)>;

REGISTER_LITE_KERNEL(reshape2, kXPU, kFloat, kNCHW, reshape2FP32, float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2, kXPU, kFP16, kNCHW, reshape2FP16, float16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2, kXPU, kFloat, kNCHW, reshape2Int32, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2, kXPU, kFloat, kNCHW, reshape2Int64, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2, kXPU, kInt8, kNCHW, reshape2INT8, int8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

using reshapeFP32 =
    paddle::lite::kernels::xpu::ReshapeCompute<float, PRECISION(kFloat)>;
using reshapeFP16 =
    paddle::lite::kernels::xpu::ReshapeCompute<float16, PRECISION(kFP16)>;
using reshapeINT8 =
    paddle::lite::kernels::xpu::ReshapeCompute<int8_t, PRECISION(kInt8)>;

REGISTER_LITE_KERNEL(reshape, kXPU, kFloat, kNCHW, reshapeFP32, float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape, kXPU, kFP16, kNCHW, reshapeFP16, float16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape, kXPU, kInt8, kNCHW, reshapeINT8, int8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .Finalize();

using flattenFP32 =
    paddle::lite::kernels::xpu::ReshapeCompute<float, PRECISION(kFloat)>;
using flattenFP16 =
    paddle::lite::kernels::xpu::ReshapeCompute<float16, PRECISION(kFP16)>;
using flattenINT8 =
    paddle::lite::kernels::xpu::ReshapeCompute<int8_t, PRECISION(kInt8)>;

REGISTER_LITE_KERNEL(flatten, kXPU, kFloat, kNCHW, flattenFP32, float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten, kXPU, kFP16, kNCHW, flattenFP16, float16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten, kXPU, kInt8, kNCHW, flattenINT8, int8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .Finalize();

using flatten2FP32 =
    paddle::lite::kernels::xpu::ReshapeCompute<float, PRECISION(kFloat)>;
using flatten2FP16 =
    paddle::lite::kernels::xpu::ReshapeCompute<float16, PRECISION(kFP16)>;

using flatten2INT8 =
    paddle::lite::kernels::xpu::ReshapeCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(flatten2, kXPU, kFloat, kNCHW, flatten2FP32, float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten2, kXPU, kFP16, kNCHW, flatten2FP16, float16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten2, kXPU, kInt8, kNCHW, flatten2INT8, int8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
