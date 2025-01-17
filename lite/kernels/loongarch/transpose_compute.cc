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

#include "lite/kernels/loongarch/transpose_compute.h"

REGISTER_LITE_KERNEL(transpose,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::TransposeCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose2,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::Transpose2Compute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::TransposeCompute<int32_t>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose2,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::Transpose2Compute<int32_t>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::TransposeCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose2,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::Transpose2Compute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kLoongArch), PRECISION(kInt64))})
    .Finalize();
