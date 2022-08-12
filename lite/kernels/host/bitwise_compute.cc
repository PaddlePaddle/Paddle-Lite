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

#include "lite/kernels/host/bitwise_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T, PrecisionType PType>
void BitwiseCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::BitwiseNotParam>();
  CHECK(param.X);
  const auto* input_data = param.X->template data<T>();
  auto* output_data = param.Out->template mutable_data<T>();
  for (int i = 0; i < param.X->numel(); ++i) {
    output_data[i] = ~input_data[i];
  }

  return;
}
}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

/*TODO: calib kernel do not support
using bitwise_bool =
    paddle::lite::kernels::host::BitwiseCompute<bool, PRECISION(kBool)>;
REGISTER_LITE_KERNEL(bitwise_not, kHost, kBool, kNCHW, bitwise_bool, bitwise_bl)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();
using bitwise_uint8 =
    paddle::lite::kernels::host::BitwiseCompute<uint8_t, PRECISION(kUInt8)>;
REGISTER_LITE_KERNEL(
    bitwise_not, kHost, kUInt8, kNCHW, bitwise_uint8, bitwise_u8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kUInt8))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kUInt8))})
    .Finalize();
using bitwise_int8 =
    paddle::lite::kernels::host::BitwiseCompute<int8_t, PRECISION(kInt8)>;
REGISTER_LITE_KERNEL(bitwise_not, kHost, kInt8, kNCHW, bitwise_int8, bitwise_i8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .Finalize();
*/
using bitwise_int16 =
    paddle::lite::kernels::host::BitwiseCompute<int16_t, PRECISION(kInt16)>;
REGISTER_LITE_KERNEL(
    bitwise_not, kHost, kInt16, kNCHW, bitwise_int16, bitwise_i16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt16))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt16))})
    .Finalize();
using bitwise_int32 =
    paddle::lite::kernels::host::BitwiseCompute<int32_t, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(
    bitwise_not, kHost, kInt32, kNCHW, bitwise_int32, bitwise_i32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
using bitwise_int64 =
    paddle::lite::kernels::host::BitwiseCompute<int64_t, PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(
    bitwise_not, kHost, kInt64, kNCHW, bitwise_int64, bitwise_i64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
