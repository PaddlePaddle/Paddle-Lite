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

#include "lite/kernels/host/argsort_compute.h"

namespace paddle {
namespace lite {
namespace kernels {}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using argsort_fp32_compute = paddle::lite::kernels::host::ArgsortCompute<float>;
REGISTER_LITE_KERNEL(
    argsort, kHost, kFloat, kAny, argsort_fp32_compute, argsort_fp32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using argsort_int32_compute =
    paddle::lite::kernels::host::ArgsortCompute<int32_t>;
REGISTER_LITE_KERNEL(
    argsort, kHost, kFloat, kAny, argsort_int32_compute, argsort_int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

using argsort_int64_compute =
    paddle::lite::kernels::host::ArgsortCompute<int64_t>;
REGISTER_LITE_KERNEL(
    argsort, kHost, kFloat, kAny, argsort_int64_compute, argsort_int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
