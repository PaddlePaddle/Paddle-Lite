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

#include "lite/kernels/host/range_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T, PrecisionType PType>
void RangeCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::RangeParam>();
  T start = (param.Start->template data<T>()[0]);
  T step = (param.Step->template data<T>()[0]);

  T* out_data = param.Out->template mutable_data<T>();
  T value = start;
  for (int i = 0; i < param.Out->dims().production(); ++i) {
    out_data[i] = value;
    value += step;
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using range_float =
    paddle::lite::kernels::host::RangeCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(range, kHost, kFloat, kAny, range_float, def)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("End",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("Step",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using range_int64 =
    paddle::lite::kernels::host::RangeCompute<int64_t, PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(range, kHost, kInt64, kAny, range_int64, def)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("End",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Step",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();

using range_int32 =
    paddle::lite::kernels::host::RangeCompute<int, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(range, kHost, kInt32, kAny, range_int32, def)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("End",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Step",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

#ifdef LITE_BUILD_EXTRA
// float kernel has higher score when picking kernel.
using range_int32_f =
    paddle::lite::kernels::host::RangeCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(range, kHost, kFloat, kAny, range_int32_f, int32)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("End",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("Step",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

// float kernel has higher score when picking kernel.
using range_int64_f =
    paddle::lite::kernels::host::RangeCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(range, kHost, kFloat, kAny, range_int64_f, int64)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("End",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("Step",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
#endif  // LITE_BUILD_EXTRA
