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

#include "lite/kernels/arm/bitwise_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using bitwise_bool = paddle::lite::kernels::arm::BitwiseCompute<bool>;
REGISTER_LITE_KERNEL(bitwise_not, kARM, kFloat, kNCHW, bitwise_bool, bitwise_bl)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kBool))})
    .Finalize();
using bitwise_uint8 = paddle::lite::kernels::arm::BitwiseCompute<uint8_t>;
REGISTER_LITE_KERNEL(
    bitwise_not, kARM, kFloat, kNCHW, bitwise_uint8, bitwise_u8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kUInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kUInt8))})
    .Finalize();
using bitwise_int8 = paddle::lite::kernels::arm::BitwiseCompute<int8_t>;
REGISTER_LITE_KERNEL(bitwise_not, kARM, kFloat, kNCHW, bitwise_int8, bitwise_i8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .Finalize();
using bitwise_int16 = paddle::lite::kernels::arm::BitwiseCompute<int16_t>;
REGISTER_LITE_KERNEL(
    bitwise_not, kARM, kFloat, kNCHW, bitwise_int16, bitwise_i16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt16))})
    .Finalize();
using bitwise_int32 = paddle::lite::kernels::arm::BitwiseCompute<int32_t>;
REGISTER_LITE_KERNEL(
    bitwise_not, kARM, kFloat, kNCHW, bitwise_int32, bitwise_i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();
using bitwise_int64 = paddle::lite::kernels::arm::BitwiseCompute<int64_t>;
REGISTER_LITE_KERNEL(
    bitwise_not, kARM, kFloat, kNCHW, bitwise_int64, bitwise_i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
