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

#include "lite/kernels/loongarch/activation_compute.h"

REGISTER_LITE_KERNEL(square,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::SquareCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(relu,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::ReluCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(leaky_relu,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::LeakyReluCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindPaddleOpVersion("leaky_relu", 1)
    .Finalize();

REGISTER_LITE_KERNEL(tanh,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::TanhCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(gelu,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::GeluCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(softsign,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::SoftsignCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(sigmoid,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::SigmoidCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(relu6,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::Relu6Compute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(sqrt,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::SqrtCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(rsqrt,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::RsqrtCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(mish,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::MishCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(hard_swish,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::HardSwishComputeCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();

REGISTER_LITE_KERNEL(erf,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::ErfCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();