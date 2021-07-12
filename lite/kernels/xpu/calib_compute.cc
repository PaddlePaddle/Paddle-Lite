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

#include "lite/kernels/xpu/calib_compute.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename InType, typename OutType>
void CalibCompute<InType, OutType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int numel = param.input->numel();
  const auto* in_data = param.input->template data<InType>();
  auto* out_data = param.output->template mutable_data<OutType>(TARGET(kXPU));
  int r = xdnn::cast_v2<InType, OutType>(
      ctx.GetRawContext(), in_data, out_data, numel);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using xpu_calib_int64_to_int32 =
    paddle::lite::kernels::xpu::CalibCompute<int64_t, int32_t>;
using xpu_calib_int32_to_int64 =
    paddle::lite::kernels::xpu::CalibCompute<int32_t, int64_t>;

REGISTER_LITE_KERNEL(
    calib, kXPU, kFloat, kNCHW, xpu_calib_int64_to_int32, int64_to_int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib, kXPU, kFloat, kNCHW, xpu_calib_int32_to_int64, int32_to_int64)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib_once, kXPU, kFloat, kNCHW, xpu_calib_int64_to_int32, int64_to_int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib_once, kXPU, kFloat, kNCHW, xpu_calib_int32_to_int64, int32_to_int64)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
