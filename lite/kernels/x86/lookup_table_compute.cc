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

#include "lite/kernels/x86/lookup_table_compute.h"

using LookupTableFloatInt64 =
    paddle::lite::kernels::x86::LookupTableCompute<float, int64_t>;
using LookupTableFloatInt32 =
    paddle::lite::kernels::x86::LookupTableCompute<float, int32_t>;

REGISTER_LITE_KERNEL(
    lookup_table, kX86, kFloat, kNCHW, LookupTableFloatInt64, def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(
    lookup_table_v2, kX86, kFloat, kNCHW, LookupTableFloatInt64, def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindPaddleOpVersion("lookup_table_v2", 1)
    .Finalize();

REGISTER_LITE_KERNEL(
    lookup_table, kX86, kFloat, kNCHW, LookupTableFloatInt32, float_int32)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(
    lookup_table_v2, kX86, kFloat, kNCHW, LookupTableFloatInt32, float_int32)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindPaddleOpVersion("lookup_table_v2", 1)
    .Finalize();
