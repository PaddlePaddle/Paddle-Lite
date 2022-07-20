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

#include "lite/kernels/xpu/gather_nd_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename DataType, typename IndexType, PrecisionType PType>
void GatherNdCompute<DataType, IndexType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto x = param.x;
  auto index = param.index;
  auto out = param.out;
  if (out->numel() == 0) {
    out->set_target(TARGET(kXPU));
    return;
  }

  std::vector<int> x_dims_cpu(x->dims().data().begin(), x->dims().data().end());
  xdnn::VectorParam<int> x_dims = xdnn::VectorParam<int>{
      x_dims_cpu.data(), static_cast<int>(x_dims_cpu.size()), nullptr};
  std::vector<int> index_dims(index->dims().data().begin(),
                              index->dims().data().end());
  int r = xdnn::gather_nd<DataType, IndexType>(
      ctx.GetRawContext(),
      x->template data<DataType>(),
      index->template data<IndexType>(),
      out->template mutable_data<DataType>(TARGET(kXPU)),
      x_dims,
      index_dims);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using GatherXPUInt32Int32 =
    paddle::lite::kernels::xpu::GatherNdCompute<int32_t,
                                                int32_t,
                                                PRECISION(kInt32)>;

using GatherXPUInt32Int64 =
    paddle::lite::kernels::xpu::GatherNdCompute<int32_t,
                                                int64_t,
                                                PRECISION(kInt32)>;

using GatherXPUFloatInt32 =
    paddle::lite::kernels::xpu::GatherNdCompute<float,
                                                int32_t,
                                                PRECISION(kFloat)>;

using GatherXPUFloatInt64 =
    paddle::lite::kernels::xpu::GatherNdCompute<float,
                                                int64_t,
                                                PRECISION(kFloat)>;

using GatherXPUInt64Int32 =
    paddle::lite::kernels::xpu::GatherNdCompute<int64_t,
                                                int32_t,
                                                PRECISION(kInt64)>;

using GatherXPUInt64Int64 =
    paddle::lite::kernels::xpu::GatherNdCompute<int64_t,
                                                int64_t,
                                                PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(gather_nd, kXPU, kFloat, kNCHW, GatherXPUFloatInt32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    gather_nd, kXPU, kFloat, kNCHW, GatherXPUFloatInt64, gather_FP32_INT64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    gather_nd, kXPU, kInt32, kNCHW, GatherXPUInt32Int32, gather_INT32_INT32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    gather_nd, kXPU, kInt32, kNCHW, GatherXPUInt32Int64, gather_INT32_INT64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    gather_nd, kXPU, kInt64, kNCHW, GatherXPUInt64Int32, gather_INT64_INT32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    gather_nd, kXPU, kInt64, kNCHW, GatherXPUInt64Int64, gather_INT64_INT64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();