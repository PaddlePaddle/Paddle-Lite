// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/transpose_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void TransposeCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x = param.x;
  auto& axis = param.axis;
  int ndims = axis.size();
  const auto x_dims = x->dims();
  std::vector<int> x_shape_host(ndims, 0);
  if (x_dims.production() == 0) {
    param.output->set_target(TARGET(kXPU));
    return;
  }

  for (int i = 0; i < ndims; ++i) {
    x_shape_host[i] = x_dims[i];
  }

  int r =
      xdnn::transpose<T>(ctx.GetRawContext(),
                         x->template data<T>(),
                         param.output->template mutable_data<T>(TARGET(kXPU)),
                         x_shape_host,
                         axis);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using transposeFP32 =
    paddle::lite::kernels::xpu::TransposeCompute<float, PRECISION(kFloat)>;
using transposeFP16 =
    paddle::lite::kernels::xpu::TransposeCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(transpose, kXPU, kFloat, kNCHW, transposeFP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose, kXPU, kFP16, kNCHW, transposeFP16, fp16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

using transpose2FP32 =
    paddle::lite::kernels::xpu::TransposeCompute<float, PRECISION(kFloat)>;
using transpose2Int32 =
    paddle::lite::kernels::xpu::TransposeCompute<int, PRECISION(kFloat)>;
using transpose2Int64 =
    paddle::lite::kernels::xpu::TransposeCompute<int64_t, PRECISION(kFloat)>;
using transpose2FP16 =
    paddle::lite::kernels::xpu::TransposeCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(transpose2, kXPU, kFloat, kNCHW, transpose2FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(
    transpose2, kXPU, kFloat, kNCHW, transpose2Int32, def_int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(
    transpose2, kXPU, kFloat, kNCHW, transpose2Int64, def_int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose2, kXPU, kFP16, kNCHW, transpose2FP16, def_fp16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
