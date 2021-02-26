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

#include "lite/kernels/host/stack_compute.h"

using stack_float32 =
    paddle::lite::kernels::host::StackCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(stack, kARM, kFloat, kAny, stack_float32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Y",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using stack_int32 =
    paddle::lite::kernels::host::StackCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(stack, kARM, kFloat, kAny, stack_int32, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Y",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

using stack_int64 =
    paddle::lite::kernels::host::StackCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(stack, kARM, kFloat, kAny, stack_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Y",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
