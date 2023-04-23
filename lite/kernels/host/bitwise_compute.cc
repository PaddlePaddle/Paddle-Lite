// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/kernels/host/elementwise_op_func.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
T naive_and(T a, T b) {
  return a & b;
}

template <class T>
T naive_or(T a, T b) {
  return a | b;
}

template <class T>
T naive_xor(T a, T b) {
  return a ^ b;
}

template <class T>
T naive_not(T a) {
  return ~a;
}

template <>
bool naive_and<bool>(bool a, bool b) {
  return a && b;
}

template <>
bool naive_or<bool>(bool a, bool b) {
  return a || b;
}

template <>
bool naive_xor<bool>(bool a, bool b) {
  return a != b;
}

template <>
bool naive_not<bool>(bool a) {
  return !a;
}

#define PROCESS_0D                                                  \
  if (param.X->dims().size() == 0 && param.Y->dims().size() == 0) { \
    auto out_ptr = param.Out->template mutable_data<T>();           \
    auto x_ptr = param.X->template data<T>();                       \
    auto y_ptr = param.Y->template data<T>();                       \
    out_ptr[0] = AndFunc(x_ptr[0], y_ptr[0]);                       \
    return;                                                         \
  }

template <typename T>
void BitwiseAndCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  CHECK(param.X);
  CHECK(param.Y);

  // ElementwiseComputeEx can do broadcasting
  std::function<T(T, T)> AndFunc = naive_and<T>;
  PROCESS_0D;
  auto batch_arg = lite::kernels::host::GenBatchElementWiseArg<T>(
      param.X, param.Y, param.Out);
  common_elmentwise_op_naive_cpu(batch_arg, AndFunc);
  return;
}

template <typename T>
void BitwiseXorCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  CHECK(param.X);
  CHECK(param.Y);

  // ElementwiseComputeEx can do broadcasting
  std::function<T(T, T)> AndFunc = naive_xor<T>;
  PROCESS_0D;
  auto batch_arg = lite::kernels::host::GenBatchElementWiseArg<T>(
      param.X, param.Y, param.Out);
  common_elmentwise_op_naive_cpu(batch_arg, AndFunc);
  return;
}

template <typename T>
void BitwiseOrCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  CHECK(param.X);
  CHECK(param.Y);

  // ElementwiseComputeEx can do broadcasting
  std::function<T(T, T)> AndFunc = naive_or<T>;
  PROCESS_0D;
  auto batch_arg = lite::kernels::host::GenBatchElementWiseArg<T>(
      param.X, param.Y, param.Out);
  common_elmentwise_op_naive_cpu(batch_arg, AndFunc);
  return;
}

template <typename T>
void BitwiseNotCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  CHECK(param.X);

  const auto* input_data = param.X->template data<T>();
  auto* out_data = param.Out->template mutable_data<T>();
  auto numel = param.X->numel();
  for (int i = 0; i < numel; ++i) {
    out_data[i] = naive_not(input_data[i]);
  }
  return;
}

#undef PROCESS_0D
}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using bitwise_and_bool = paddle::lite::kernels::host::BitwiseAndCompute<bool>;
REGISTER_LITE_KERNEL(bitwise_and, kHost, kAny, kNCHW, bitwise_and_bool, bl)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();

using bitwise_and_int32_t =
    paddle::lite::kernels::host::BitwiseAndCompute<int32_t>;
REGISTER_LITE_KERNEL(
    bitwise_and, kHost, kAny, kNCHW, bitwise_and_int32_t, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

using bitwise_and_int64_t =
    paddle::lite::kernels::host::BitwiseAndCompute<int64_t>;
REGISTER_LITE_KERNEL(
    bitwise_and, kHost, kAny, kNCHW, bitwise_and_int64_t, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

using bitwise_not_bool = paddle::lite::kernels::host::BitwiseNotCompute<bool>;
REGISTER_LITE_KERNEL(bitwise_not, kHost, kAny, kNCHW, bitwise_not_bool, bl)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();

using bitwise_not_int32_t =
    paddle::lite::kernels::host::BitwiseNotCompute<int32_t>;
REGISTER_LITE_KERNEL(
    bitwise_not, kHost, kAny, kNCHW, bitwise_not_int32_t, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

using bitwise_not_int64_t =
    paddle::lite::kernels::host::BitwiseNotCompute<int64_t>;
REGISTER_LITE_KERNEL(
    bitwise_not, kHost, kAny, kNCHW, bitwise_not_int64_t, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

#ifdef LITE_BUILD_EXTRA
using bitwise_xor_bool = paddle::lite::kernels::host::BitwiseXorCompute<bool>;
REGISTER_LITE_KERNEL(bitwise_xor, kHost, kAny, kNCHW, bitwise_xor_bool, bl)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();

using bitwise_xor_int32_t =
    paddle::lite::kernels::host::BitwiseXorCompute<int32_t>;
REGISTER_LITE_KERNEL(
    bitwise_xor, kHost, kAny, kNCHW, bitwise_xor_int32_t, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

using bitwise_xor_int64_t =
    paddle::lite::kernels::host::BitwiseXorCompute<int64_t>;
REGISTER_LITE_KERNEL(
    bitwise_xor, kHost, kAny, kNCHW, bitwise_xor_int64_t, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

using bitwise_or_bool = paddle::lite::kernels::host::BitwiseOrCompute<bool>;
REGISTER_LITE_KERNEL(bitwise_or, kHost, kAny, kNCHW, bitwise_or_bool, bl)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();

using bitwise_or_int32_t =
    paddle::lite::kernels::host::BitwiseOrCompute<int32_t>;
REGISTER_LITE_KERNEL(bitwise_or, kHost, kAny, kNCHW, bitwise_or_int32_t, int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

using bitwise_or_int64_t =
    paddle::lite::kernels::host::BitwiseOrCompute<int64_t>;
REGISTER_LITE_KERNEL(bitwise_or, kHost, kAny, kNCHW, bitwise_or_int64_t, int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
#endif  // LITE_BUILD_EXTRA
