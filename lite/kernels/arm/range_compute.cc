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
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

using range_float =
    paddle::lite::kernels::host::RangeCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(range, kARM, kFloat, kNCHW, range_float, def)
    .BindInput("Start", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("End", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Step", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

using range_int32 =
    paddle::lite::kernels::host::RangeCompute<int, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(range, kARM, kInt32, kNCHW, range_int32, def)
    .BindInput("Start",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("End", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Step", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .Finalize();
