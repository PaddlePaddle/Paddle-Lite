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

template <typename InType, typename OutType, PrecisionType PType>
void CalibCompute<InType, OutType, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  if (param.scale) {
    std::vector<float> cpu_scale = {param.scale};
    calib_max_guard_ = TargetWrapperXPU::MallocScratchPad(sizeof(float));
    lite::TargetWrapperXPU::MemcpySync(calib_max_guard_->addr_,
                                       cpu_scale.data(),
                                       sizeof(float),
                                       IoDirection::HtoD);
  }
}

template <typename InType, typename OutType, PrecisionType PType>
void CalibCompute<InType, OutType, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  int numel = param.input->numel();
  const auto* in_data = param.input->template data<InType>();
  auto* out_data = param.output->template mutable_data<OutType>(TARGET(kXPU));
  if (numel == 0) {
    return;
  }
  int r = xdnn::cast_v2<InType, OutType>(
      ctx.GetRawContext(), in_data, out_data, numel);
  CHECK_EQ(r, 0);
}

template <>
void CalibCompute<float, int8_t, PRECISION(kInt8)>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  CHECK(param.scale);
  int numel = param.input->numel();
  const auto* in_data = param.input->template data<float>();
  auto* out_data = param.output->template mutable_data<int8_t>(TARGET(kXPU));
  if (numel == 0) {
    return;
  }

  int r = xdnn::quantization<float, int8_t>(
      ctx.GetRawContext(),
      in_data,
      out_data,
      numel,
      reinterpret_cast<const float*>(calib_max_guard_->addr_));
  CHECK_EQ(r, 0);
}

template <>
void CalibCompute<int8_t, float, PRECISION(kInt8)>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  CHECK(param.scale);
  int numel = param.input->numel();
  const auto* in_data = param.input->template data<int8_t>();
  auto* out_data = param.output->template mutable_data<float>(TARGET(kXPU));
  if (numel == 0) {
    return;
  }

  int r = xdnn::dequantization<int8_t, float>(
      ctx.GetRawContext(),
      in_data,
      out_data,
      numel,
      reinterpret_cast<const float*>(calib_max_guard_->addr_));
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using xpu_calib_int64_to_int32 =
    paddle::lite::kernels::xpu::CalibCompute<int64_t,
                                             int32_t,
                                             PRECISION(kFloat)>;
using xpu_calib_int32_to_int64 =
    paddle::lite::kernels::xpu::CalibCompute<int32_t,
                                             int64_t,
                                             PRECISION(kFloat)>;
using xpu_calib_fp32_to_fp16 =
    paddle::lite::kernels::xpu::CalibCompute<float, float16, PRECISION(kFloat)>;
using xpu_calib_fp16_to_fp32 =
    paddle::lite::kernels::xpu::CalibCompute<float16, float, PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(
    calib, kXPU, kFloat, kNCHW, xpu_calib_int64_to_int32, calib_int64_to_int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib, kXPU, kFloat, kNCHW, xpu_calib_int32_to_int64, calib_int32_to_int64)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib, kXPU, kFloat, kNCHW, xpu_calib_fp32_to_fp16, calib_fp32_to_fp16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib, kXPU, kFloat, kNCHW, xpu_calib_fp16_to_fp32, calib_fp16_to_fp32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(calib_once,
                     kXPU,
                     kFloat,
                     kNCHW,
                     xpu_calib_int64_to_int32,
                     calib_int64_to_int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(calib_once,
                     kXPU,
                     kFloat,
                     kNCHW,
                     xpu_calib_int32_to_int64,
                     calib_int32_to_int64)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib_once, kXPU, kFloat, kNCHW, xpu_calib_fp32_to_fp16, calib_fp32_to_fp16)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib_once, kXPU, kFloat, kNCHW, xpu_calib_fp16_to_fp32, calib_fp16_to_fp32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

using xpu_calib_fp32_to_int8 =
    paddle::lite::kernels::xpu::CalibCompute<float, int8_t, PRECISION(kInt8)>;

using xpu_calib_int8_to_fp32 =
    paddle::lite::kernels::xpu::CalibCompute<int8_t, float, PRECISION(kInt8)>;
REGISTER_LITE_KERNEL(
    calib, kXPU, kInt8, kNCHW, xpu_calib_fp32_to_int8, calib_fp32_to_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib, kXPU, kInt8, kNCHW, xpu_calib_int8_to_fp32, calib_int8_to_fp32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();
