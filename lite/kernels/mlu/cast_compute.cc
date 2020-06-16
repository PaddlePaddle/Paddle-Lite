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

#include "lite/kernels/mlu/cast_compute.h"
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

REGISTER_LITE_KERNEL(cast,
                     kMLU,
                     kFloat,
                     kNHWC,
                     paddle::lite::kernels::mlu::CastFp32toFp16,
                     fp32_to_fp16)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMLU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMLU),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(cast,
                     kMLU,
                     kFloat,
                     kNHWC,
                     paddle::lite::kernels::mlu::CastFp16toFp32,
                     fp16_to_fp32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kMLU),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kMLU),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();
