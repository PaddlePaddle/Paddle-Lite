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

#include "lite/kernels/host/increment_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
void increment(const T* input, const int n, const T step, T* out) {
  for (int i = 0; i < n; i++) {
    out[i] = input[i] + step;
  }
}

template <typename T, PrecisionType PType>
void IncrementCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::IncrementParam>();
  int total_num = param.X->numel();
  const auto* x_data = param.X->template data<T>();
  auto* o_data = param.Out->template mutable_data<T>();
  T step = static_cast<T>(param.step);
  increment<T>(x_data, total_num, step, o_data);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using increment_float32 =
    paddle::lite::kernels::host::IncrementCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(increment, kHost, kFloat, kAny, increment_float32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .Finalize();

using increment_int32 =
    paddle::lite::kernels::host::IncrementCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(increment, kHost, kFloat, kAny, increment_int32, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kAny), -1)})
    .Finalize();

using increment_int64 =
    paddle::lite::kernels::host::IncrementCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(increment, kHost, kFloat, kAny, increment_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt64), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kInt64), DATALAYOUT(kAny), -1)})
    .Finalize();
