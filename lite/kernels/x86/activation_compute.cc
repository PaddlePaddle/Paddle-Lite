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

#include "lite/kernels/x86/activation_compute.h"

// float
REGISTER_LITE_KERNEL(square,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::SquareCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

// float
REGISTER_LITE_KERNEL(relu,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::ReluCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

// float
REGISTER_LITE_KERNEL(leaky_relu,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::LeakyReluCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

// float
REGISTER_LITE_KERNEL(tanh,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::TanhCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

// float
REGISTER_LITE_KERNEL(gelu,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::GeluCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(softsign,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::SoftsignCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

// float
REGISTER_LITE_KERNEL(sigmoid,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::SigmoidCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

// float
REGISTER_LITE_KERNEL(relu6,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::Relu6Compute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
