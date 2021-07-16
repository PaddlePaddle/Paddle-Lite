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
#include "lite/backends/arm/math/fp16/conv_transpose_depthwise_fp16.h"
#include "lite/backends/arm/math/fp16/elementwise_fp16.h"
#include "lite/backends/arm/math/fp16/fill_bias_act_fp16.h"
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
template <lite_api::ActivationType Act = lite_api::ActivationType::kIndentity>
inline float16x8_t vactive_f16(const float16x8_t& x) {
  return x;
}

template <>
inline float16x8_t vactive_f16<lite_api::ActivationType::kRelu>(
    const float16x8_t& x) {
  float16x8_t __zero = vdupq_n_f16(0.f);
  return vmaxq_f16(x, __zero);
}

template <>
inline float16x8_t vactive_f16<lite_api::ActivationType::kRelu6>(
    const float16x8_t& x) {
  float16x8_t __zero = vdupq_n_f16(0.f);
  float16x8_t __six = vdupq_n_f16(6.f);
  return vminq_f16(vmaxq_f16(x, __zero), __six);
}

template <>
inline float16x8_t vactive_f16<lite_api::ActivationType::kSigmoid>(
    const float16x8_t& x) {
  float16x8_t __one = vdupq_n_f16(1.f);
  float16x8_t __x = vnegq_f16(x);
  __x = expq_ps_f16(__x);
  __x = vaddq_f16(__x, __one);
  float16x8_t __out = vrecpeq_f16(__x);
  return vmulq_f16(vrecpsq_f16(__x, __out), __out);
}

template <>
inline float16x8_t vactive_f16<lite_api::ActivationType::kTanh>(
    const float16x8_t& x) {
  float16x8_t __one = vdupq_n_f16(1.f);
  float16x8_t __x = vmulq_n_f16(x, -2.f);
  __x = expq_ps_f16(__x);
  __x = vaddq_f16(__x, __one);
  float16x8_t __out = vrecpeq_f16(__x);
  __out = vmulq_f16(vrecpsq_f16(__x, __out), __out);
  __out = vmulq_n_f16(__out, 2.f);
  return vsubq_f16(__out, __one);
}

template <lite_api::ActivationType Act = lite_api::ActivationType::kIndentity>
inline float16_t active_f16(const float16_t& x) {
  return x;
}

template <>
inline float16_t active_f16<lite_api::ActivationType::kRelu>(
    const float16_t& x) {
  return (x > 0.f ? x : 0.f);
}

template <>
inline float16_t active_f16<lite_api::ActivationType::kRelu6>(
    const float16_t& x) {
  float16_t max_val = (x > 0.f ? x : 0.f);
  return max_val > 6.f ? 6.f : max_val;
}

template <>
inline float16_t active_f16<lite_api::ActivationType::kSigmoid>(
    const float16_t& x) {
  return 1.f / (1.f + exp(-x));
}

template <>
inline float16_t active_f16<lite_api::ActivationType::kTanh>(
    const float16_t& x) {
  return 2.f / (1.f + exp(-2.f * x)) - 1.f;
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
