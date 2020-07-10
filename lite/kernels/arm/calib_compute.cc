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

#include "lite/kernels/arm/calib_compute.h"
#include <vector>
#include "lite/backends/arm/math/type_trans.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <DataLayoutType DLType>
void CalibComputeFp32ToInt8<DLType>::Run() {
  auto& param = this->template Param<operators::CalibParam>();
  std::vector<float> scale = {param.scale};
  const auto* din = param.input->template data<float>();
  auto* dout = param.output->template mutable_data<signed char>();
  lite::arm::math::fp32_to_int8(
      din, dout, scale.data(), 1, 1, param.input->numel());
}

template <DataLayoutType DLType>
void CalibComputeInt64ToInt32<DLType>::Run() {
  auto& param = this->template Param<operators::CalibParam>();
  const auto* din = param.input->template data<int64_t>();
  std::vector<float> scale = {param.scale};
  auto* dout = param.output->template mutable_data<int32_t>();
  for (auto i = 0; i < param.input->numel(); ++i) {
    dout[i] = din[i];
  }
}

template <DataLayoutType DLType>
void CalibComputeInt8ToFp32<DLType>::Run() {
  auto& param = this->template Param<operators::CalibParam>();
  const auto* din = param.input->template data<signed char>();
  std::vector<float> scale = {param.scale};
  auto* dout = param.output->template mutable_data<float>();
  lite::arm::math::int8_to_fp32(
      din, dout, scale.data(), 1, 1, param.input->numel());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    calib,
    kARM,
    kInt8,
    kNCHW,
    paddle::lite::kernels::arm::CalibComputeFp32ToInt8<DATALAYOUT(kNCHW)>,
    fp32_to_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib,
    kARM,
    kInt8,
    kNCHW,
    paddle::lite::kernels::arm::CalibComputeInt8ToFp32<DATALAYOUT(kNCHW)>,
    int8_to_fp32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib,
    kARM,
    kInt8,
    kNHWC,
    paddle::lite::kernels::arm::CalibComputeFp32ToInt8<DATALAYOUT(kNHWC)>,
    fp32_to_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib,
    kARM,
    kInt8,
    kNHWC,
    paddle::lite::kernels::arm::CalibComputeInt8ToFp32<DATALAYOUT(kNHWC)>,
    int8_to_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib,
    kARM,
    kInt64,
    kNCHW,
    paddle::lite::kernels::arm::CalibComputeInt64ToInt32<DATALAYOUT(kNCHW)>,
    int64_to_int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib_once,
    kARM,
    kInt8,
    kNCHW,
    paddle::lite::kernels::arm::CalibComputeFp32ToInt8<DATALAYOUT(kNCHW)>,
    fp32_to_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib_once,
    kARM,
    kInt8,
    kNCHW,
    paddle::lite::kernels::arm::CalibComputeInt8ToFp32<DATALAYOUT(kNCHW)>,
    int8_to_fp32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib_once,
    kARM,
    kInt8,
    kNHWC,
    paddle::lite::kernels::arm::CalibComputeFp32ToInt8<DATALAYOUT(kNHWC)>,
    fp32_to_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib_once,
    kARM,
    kInt8,
    kNHWC,
    paddle::lite::kernels::arm::CalibComputeInt8ToFp32<DATALAYOUT(kNHWC)>,
    int8_to_fp32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    calib_once,
    kARM,
    kInt64,
    kNCHW,
    paddle::lite::kernels::arm::CalibComputeInt64ToInt32<DATALAYOUT(kNCHW)>,
    int64_to_int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
