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

#include "lite/kernels/loongarch/batch_norm_compute.h"

REGISTER_LITE_KERNEL(batch_norm,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::BatchNormCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(sync_batch_norm,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::BatchNormCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();
