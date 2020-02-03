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

#include "lite/kernels/host/increment_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <PrecisionType PType, typename DType>
void IncrementCompute<PType, DType>::Run() {
  auto& param = this->template Param<param_t>();
  DType step = static_cast<DType>(param.step);
  int num = param.X->dims().production();
  const auto* x_data = param.X->template data<DType>();
  auto* o_data = param.Out->template mutable_data<DType>();
  for (int i = 0; i < num; i++) {
    o_data[i] = x_data[i] + step;
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using IncrementCompute_FP32 =
    paddle::lite::kernels::host::IncrementCompute<PRECISION(kFloat), float>;

using IncrementCompute_INT64 =
    paddle::lite::kernels::host::IncrementCompute<PRECISION(kInt64), int64_t>;

REGISTER_LITE_KERNEL(increment, kHost, kFloat, kAny, IncrementCompute_FP32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kAny), -1)})
    .Finalize();

REGISTER_LITE_KERNEL(
    increment, kHost, kInt64, kAny, IncrementCompute_INT64, def)
    .BindInput("X",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kInt64), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kInt64), DATALAYOUT(kAny), -1)})
    .Finalize();
