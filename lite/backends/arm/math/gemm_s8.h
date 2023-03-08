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
#include "lite/backends/arm/math/gemm_prepacked_int8.h"
#include "lite/backends/arm/math/gemv_arm_int8.h"
#include "lite/core/context.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename Dtype>
void gemm_s8(bool is_transA,
             bool is_transB,
             bool packed_b,
             int M,
             int N,
             int K,
             const int8_t* A,
             const int8_t* B,
             Dtype* C,
             const float* bias,
             bool is_bias,
             GemmBiasDirection bias_direction,
             const float* scale,
             const operators::ActivationParam act_param,
             ARMContext* ctx);

#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
template <typename Dtype>
void gemm_sve(bool is_transA,
              bool is_transB,
              int M,
              int N,
              int K,
              const int8_t* A,
              const int8_t* B,
              Dtype* C,
              const float* bias,
              bool is_bias,
              GemmBiasDirection bias_direction,
              const float* scale,
              const operators::ActivationParam act_param,
              ARMContext* ctx);
#endif
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
