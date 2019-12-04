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

#include "lite/kernels/bm/calib_compute.h"
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace bm {

void CalibComputeFp32ToInt8::Run() {
}

void CalibComputeInt8ToFp32::Run() {
  return;
}

}  // namespace bm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(calib,
                     kBM,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::bm::CalibComputeFp32ToInt8,
                     fp32_to_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kBM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kBM), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(calib,
                     kBM,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::bm::CalibComputeInt8ToFp32,
                     int8_to_fp32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kBM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kBM), PRECISION(kFloat))})
    .Finalize();
REGISTER_LITE_KERNEL(calib_once,
                     kBM,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::bm::CalibComputeFp32ToInt8,
                     fp32_to_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kBM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kBM), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(calib_once,
                     kBM,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::bm::CalibComputeInt8ToFp32,
                     int8_to_fp32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kBM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kBM), PRECISION(kFloat))})
    .Finalize();
