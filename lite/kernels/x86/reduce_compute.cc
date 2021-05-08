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

#include "lite/kernels/x86/reduce_compute.h"

REGISTER_LITE_KERNEL(reduce_sum,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ReduceSumCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_prod,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ReduceProdCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_prod,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ReduceProdCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_prod,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ReduceProdCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_mean,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ReduceMeanCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(reduce_max,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ReduceMaxCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
