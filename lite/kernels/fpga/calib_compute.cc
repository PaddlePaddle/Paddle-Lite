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

#include "lite/kernels/fpga/calib_compute.h"
#include <vector>
#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {
using float16 = zynqmp::float16;

void CalibComputeFp32ToFP16::Run() {
  auto& param = this->Param<operators::CalibParam>();
  const auto* din = param.input->data<float>();
  param.output->mutable_data<float16>();
  param.output->ZynqTensor()->copyFrom(param.input->ZynqTensor());
  auto out_lod = param.output->mutable_lod();
  *out_lod = param.input->lod();
  return;
}

void CalibComputeFP16ToFp32::Run() {
  auto& param = this->Param<operators::CalibParam>();
  const auto* din = param.input->data<float16>();
  auto* dout = param.output->mutable_data<float>();
  param.output->ZynqTensor()->copyFrom(param.input->ZynqTensor());
  auto out_lod = param.output->mutable_lod();
  *out_lod = param.input->lod();
  return;
}

void CalibComputeFloat2Int::Run() {
  auto& param = this->Param<operators::CalibParam>();
  const auto* din = param.input->data<float>();
  auto* dout = param.output->mutable_data<int>();

  for (int i = 0; i < param.input->numel(); i++) {
    dout[i] = static_cast<float>(din[i]);
  }
  auto out_lod = param.output->mutable_lod();
  *out_lod = param.input->lod();
  return;
}

void CalibComputeInt64ToFloat::Run() {
  auto& param = this->Param<operators::CalibParam>();
  const auto* din = param.input->data<int64_t>();
  auto* dout = param.output->mutable_data<float>();
  for (int i = 0; i < param.input->numel(); i++) {
    dout[i] = static_cast<float>(din[i]);
  }

  auto out_lod = param.output->mutable_lod();
  *out_lod = param.input->lod();
  return;
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(calib,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::CalibComputeFp32ToFP16,
                     fp32_to_fp16_fpga)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(calib_once,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::CalibComputeFp32ToFP16,
                     fp32_to_fp16_fpga)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(calib,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::CalibComputeFloat2Int,
                     float_2_int_fpga)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(calib_once,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::CalibComputeFloat2Int,
                     float_2_int_fpga)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(calib,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::CalibComputeFP16ToFp32,
                     float_to_int_fpga)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(calib_once,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::CalibComputeFP16ToFp32,
                     fp16_to_fp32_fpga)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(calib,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::CalibComputeInt64ToFloat,
                     int64_2_float_fpga)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(calib_once,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::CalibComputeInt64ToFloat,
                     int64_2_float_fpga)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();
