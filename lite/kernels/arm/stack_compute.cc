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
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

using stack_float =
    paddle::lite::kernels::host::StackCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(stack, kARM, kFloat, kNCHW, stack_float, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using stack_int32 =
    paddle::lite::kernels::host::StackCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(stack, kARM, kFloat, kNCHW, stack_int32, int32_def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();
