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

#include "lite/kernels/x86/gru_compute.h"
#include "lite/utils/env.h"

// DEFINE_int32(paddle_num_threads,
//              1,
//              "Number of threads for each paddle instance.");
int32_t paddle_num_threads =
    paddle::lite::GetIntFromEnv("paddle_num_threads", 1);

REGISTER_LITE_KERNEL(gru,
                     kX86,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::x86::GRUCompute<float>,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("BatchResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("BatchHidden", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
