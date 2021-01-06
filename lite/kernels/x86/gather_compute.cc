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

#include "lite/kernels/x86/gather_compute.h"

REGISTER_LITE_KERNEL(gather, kX86, kFloat, kNCHW, GatherInt32Int32, int32int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Axis", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(gather, kX86, kFloat, kNCHW, GatherInt64Int64, int64int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Axis", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kAny))})
    .Finalize();
REGISTER_LITE_KERNEL(gather, kX86, kFloat, kNCHW, GatherInt64Int32, int64int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindInput("Axis", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kAny))})
    .Finalize();
REGISTER_LITE_KERNEL(gather, kX86, kFloat, kNCHW, GatherInt32Int64, int32int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("Axis", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kAny))})
    .Finalize();
