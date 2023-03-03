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
#include <algorithm>
#include "lite/kernels/host/elementwise_op_func.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
struct BitwiseNotFunctor {
  T operator()(const T a) const { return ~a; }
};

template <>
struct BitwiseNotFunctor<bool> {
  bool operator()(const bool a) const { return !a; }
};

template <typename T>
void BitwiseAndCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  CHECK(param.X);
  CHECK(param.Y);

  // ElementwiseComputeEx can do broadcasting
  std::function<T(T, T)> AndFunc = naive_and<T>;
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
  BitwiseNotFunctor<T> func;
  std::transform(input_data, input_data + numel, out_data, func);
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using bitwise_and_bool = paddle::lite::kernels::host::BitwiseAndCompute<bool>;
REGISTER_LITE_KERNEL(bitwise_and, kHost, kBool, kNCHW, bitwise_and_bool, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();

using bitwise_and_int32_t =
    paddle::lite::kernels::host::BitwiseAndCompute<int32_t>;
REGISTER_LITE_KERNEL(
    bitwise_and, kHost, kInt32, kNCHW, bitwise_and_int32_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

using bitwise_and_int64_t =
    paddle::lite::kernels::host::BitwiseAndCompute<int64_t>;
REGISTER_LITE_KERNEL(
    bitwise_and, kHost, kInt64, kNCHW, bitwise_and_int64_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

using bitwise_not_bool = paddle::lite::kernels::host::BitwiseNotCompute<bool>;
REGISTER_LITE_KERNEL(bitwise_not, kHost, kBool, kNCHW, bitwise_not_bool, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();

using bitwise_not_int32_t =
    paddle::lite::kernels::host::BitwiseNotCompute<int32_t>;
REGISTER_LITE_KERNEL(
    bitwise_not, kHost, kInt32, kNCHW, bitwise_not_int32_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

using bitwise_not_int64_t =
    paddle::lite::kernels::host::BitwiseNotCompute<int64_t>;
REGISTER_LITE_KERNEL(
    bitwise_not, kHost, kInt64, kNCHW, bitwise_not_int64_t, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
