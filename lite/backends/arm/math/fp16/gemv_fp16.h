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

#pragma once

#include <cmath>
#include "lite/backends/arm/math/fp16/common_preprocess.h"
#include "lite/core/context.h"
#include "lite/core/device_info.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
typedef __fp16 float16_t;
void gemv_fp16(const float16_t *A,
               const float16_t *x,
               float16_t *y,
               bool transA,
               int M,
               int N,
               float16_t beta,
               bool is_bias,
               const float16_t *bias,
               bool flag_act,
               const operators::ActivationParam act_param,
               ARMContext *ctx);
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
