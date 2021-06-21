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

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include "lite/backends/arm/math/fp16/activation_fp16.h"
#include "lite/backends/arm/math/fp16/conv3x3_depthwise_fp16.h"
#include "lite/backends/arm/math/fp16/conv_block_utils_fp16.h"
#include "lite/backends/arm/math/fp16/conv_impl_fp16.h"
#include "lite/backends/arm/math/fp16/elementwise_fp16.h"
#include "lite/backends/arm/math/fp16/gemm_c8_fp16.h"
#include "lite/backends/arm/math/fp16/gemm_fp16.h"
#include "lite/backends/arm/math/fp16/gemv_fp16.h"
#include "lite/backends/arm/math/fp16/interpolate_fp16.h"
#include "lite/backends/arm/math/fp16/pad2d_fp16.h"
#include "lite/backends/arm/math/fp16/pooling_fp16.h"
#include "lite/backends/arm/math/fp16/sgemm_fp16.h"
#include "lite/backends/arm/math/fp16/shuffle_channel_fp16.h"
#include "lite/backends/arm/math/fp16/softmax_fp16.h"
#include "lite/backends/arm/math/fp16/type_trans_fp16.h"
#include "lite/backends/arm/math/funcs.h"
typedef __fp16 float16_t;

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

template <typename T>
void fill_bias_fc(
    T* tensor, const T* bias, int num, int channel, bool flag_relu);

// exp() computed for 8 float at once
inline float16x8_t expq_ps_f16(float16x8_t x) {
  float32x4_t va = vcvt_f32_f16(vget_high_f16(x));
  float32x4_t vb = vcvt_f32_f16(vget_low_f16(x));
  float32x4_t vexpa = exp_ps(va);
  float32x4_t vexpb = exp_ps(vb);
  float16x8_t vres;
  float16x4_t vresa = vcvt_f16_f32(vexpa);
  float16x4_t vresb = vcvt_f16_f32(vexpb);
  for (int i = 0; i < 3; i++) {
    vres[i + 4] = vresa[i];
  }
  for (int i = 0; i < 3; i++) {
    vres[i] = vresb[i];
  }
  return vres;
}

inline float16x4_t exp_ps_f16(float16x4_t x) {
  float32x4_t va = vcvt_f32_f16(x);
  float32x4_t vexpa = exp_ps(va);
  float16x4_t vresa = vcvt_f16_f32(vexpa);
  return vresa;
}

inline float16x8_t divq_ps_f16(float16x8_t a, float16x8_t b) {
  float16x8_t reciprocal = vrecpeq_f16(b);
  reciprocal = vmulq_f16(vrecpsq_f16(b, reciprocal), reciprocal);
  return vmulq_f16(a, reciprocal);
}

inline float16x4_t div_ps_f16(float16x4_t a, float16x4_t b) {
  float16x4_t reciprocal = vrecpe_f16(b);
  reciprocal = vmul_f16(vrecps_f16(b, reciprocal), reciprocal);
  return vmul_f16(a, reciprocal);
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
