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

#include "lite/kernels/xpu/gru_compute.h"

REGISTER_LITE_KERNEL(
    gru, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::GRUCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("BatchResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("BatchHidden", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
