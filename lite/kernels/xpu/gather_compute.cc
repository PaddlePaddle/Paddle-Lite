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

#include "lite/kernels/xpu/gather_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename DataType, typename IndexType>
void GatherCompute<DataType, IndexType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto x = param.X;
  auto index = param.Index;
  auto out = param.Out;
  if (out->numel() == 0) {
    out->set_target(TARGET(kXPU));
    return;
  }
  int axis = 0;
  if (param.Axis != nullptr) {
    CHECK(param.Axis->precision() == PRECISION(kInt32))
        << " xpu only support axis int32 type";
    auto* axis_data = param.Axis->template data<int>();
    axis = axis_data[0];
  }
  std::vector<int> x_dims(x->dims().data().begin(), x->dims().data().end());
  if (axis < 0) {
    axis += x_dims.size();
  }

  if (param.X->precision() == PrecisionType::kInt64 &&
      param.Index->precision() == PrecisionType::kInt64) {
    auto* index_int64 = param.Index->template data<int64_t>();
    int size = param.Index->dims().production();
    XPUScratchPadGuard index_xpu_guard_ =
        TargetWrapperXPU::MallocScratchPad(size * sizeof(int));
    int* index_int32_device = reinterpret_cast<int*>(index_xpu_guard_->addr_);

    int r0 = xdnn::cast_v2<int64_t, int32_t>(
        ctx.GetRawContext(), index_int64, index_int32_device, index->numel());
    CHECK_EQ(r0, 0);

    int r1 = xdnn::gather<int64_t, int32_t>(
        ctx.GetRawContext(),
        x->template data<int64_t>(),
        index_int32_device,
        out->template mutable_data<int64_t>(TARGET(kXPU)),
        x_dims,
        index->numel(),
        axis);
    CHECK_EQ(r1, 0);
  } else if (param.X->precision() == PrecisionType::kInt64 &&
             param.Index->precision() == PrecisionType::kInt32) {
    int r = xdnn::gather<int64_t, int32_t>(
        ctx.GetRawContext(),
        x->template data<int64_t>(),
        index->template data<int32_t>(),
        out->template mutable_data<int64_t>(TARGET(kXPU)),
        x_dims,
        index->numel(),
        axis);
    CHECK_EQ(r, 0);
  } else if (param.X->precision() == PrecisionType::kInt32 &&
             param.Index->precision() == PrecisionType::kInt32) {
    int r = xdnn::gather<int32_t, int32_t>(
        ctx.GetRawContext(),
        x->template data<int32_t>(),
        index->template data<int32_t>(),
        out->template mutable_data<int32_t>(TARGET(kXPU)),
        x_dims,
        index->numel(),
        axis);
    CHECK_EQ(r, 0);
  } else if (param.X->precision() == PrecisionType::kInt32 &&
             param.Index->precision() == PrecisionType::kInt64) {
    int r = xdnn::gather<int32_t, int64_t>(
        ctx.GetRawContext(),
        x->template data<int32_t>(),
        index->template data<int64_t>(),
        out->template mutable_data<int32_t>(TARGET(kXPU)),
        x_dims,
        index->numel(),
        axis);
    CHECK_EQ(r, 0);
  } else if (param.X->precision() == PrecisionType::kFloat &&
             param.Index->precision() == PrecisionType::kInt32) {
    int r = xdnn::gather<float, int32_t>(
        ctx.GetRawContext(),
        x->template data<float>(),
        index->template data<int32_t>(),
        out->template mutable_data<float>(TARGET(kXPU)),
        x_dims,
        index->numel(),
        axis);
    CHECK_EQ(r, 0);
  } else if (param.X->precision() == PrecisionType::kFloat &&
             param.Index->precision() == PrecisionType::kInt64) {
    int r = xdnn::gather<float, int64_t>(
        ctx.GetRawContext(),
        x->template data<float>(),
        index->template data<int64_t>(),
        out->template mutable_data<float>(TARGET(kXPU)),
        x_dims,
        index->numel(),
        axis);
    CHECK_EQ(r, 0);
  } else {
    LOG(FATAL) << "Unsupported gather op with x dtype: "
               << lite_api::PrecisionToStr(param.X->precision())
               << " and index dtype: "
               << lite_api::PrecisionToStr(param.Index->precision());
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(gather, kXPU, kFloat, kNCHW, GatherXPUFloatInt32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(
    gather, kXPU, kFloat, kNCHW, GatherXPUFloatInt64, gather_float_i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(
    gather, kXPU, kFloat, kNCHW, GatherXPUInt32Int32, gather_i32_i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(
    gather, kXPU, kFloat, kNCHW, GatherXPUInt32Int64, gather_i32_i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(
    gather, kXPU, kFloat, kNCHW, GatherXPUInt64Int32, gather_i64_i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
REGISTER_LITE_KERNEL(
    gather, kXPU, kFloat, kNCHW, GatherXPUInt64Int64, gather_i64_i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
