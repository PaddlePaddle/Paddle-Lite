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

#include "lite/kernels/arm/sum_compute.h"

#include <string>

#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using sum_int32 = paddle::lite::kernels::arm::SumCompute<int32_t>;
REGISTER_LITE_KERNEL(sum, kARM, kFloat, kNCHW, sum_int32, sum_i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();

using sum_int64 = paddle::lite::kernels::arm::SumCompute<int64_t>;
REGISTER_LITE_KERNEL(sum, kARM, kFloat, kNCHW, sum_int64, sum_i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();

using sum_fp32 = paddle::lite::kernels::arm::SumCompute<float>;
REGISTER_LITE_KERNEL(sum, kARM, kFloat, kNCHW, sum_fp32, sum_fp32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
