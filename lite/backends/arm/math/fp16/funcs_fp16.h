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
#include "lite/backends/arm/math/fp16/power_fp16.h"
#include "lite/backends/arm/math/fp16/sgemm_fp16.h"
#include "lite/backends/arm/math/fp16/softmax_fp16.h"
#include "lite/backends/arm/math/fp16/sparse_conv_fp16.h"
#include "lite/backends/arm/math/fp16/sparse_semi_conv_fp16.h"
#include "lite/backends/arm/math/fp16/type_trans_fp16.h"
#include "lite/backends/arm/math/funcs.h"
typedef __fp16 float16_t;

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

template <typename T>
void fill_bias_fc(T* tensor,
                  const T* bias,
                  int num,
                  int channel,
                  const operators::ActivationParam* act_param);

static inline float16x4_t reciprocal_ps_f16(const float16x4_t& x) {
  float16x4_t _reciprocal = vrecpe_f16(x);
  _reciprocal = vmul_f16(vrecps_f16(x, _reciprocal), _reciprocal);
  _reciprocal = vmul_f16(vrecps_f16(x, _reciprocal), _reciprocal);
  return _reciprocal;
}

static inline float16x8_t reciprocalq_ps_f16(const float16x8_t& x) {
  float16x8_t _reciprocal = vrecpeq_f16(x);
  _reciprocal = vmulq_f16(vrecpsq_f16(x, _reciprocal), _reciprocal);
  _reciprocal = vmulq_f16(vrecpsq_f16(x, _reciprocal), _reciprocal);
  return _reciprocal;
}

#define c_exp_hi_f16 10.7421875f
#define c_exp_lo_f16 -10.7421875f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
static inline float16x4_t exp_ps_naive(float16x4_t x) {
  float16x4_t tmp, fx;

  float16x4_t one = vdup_n_f16(1);
  x = vmin_f16(x, vdup_n_f16(c_exp_hi_f16));
  x = vmax_f16(x, vdup_n_f16(c_exp_lo_f16));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = vfma_f16(vdup_n_f16(0.5f), x, vdup_n_f16(c_cephes_LOG2EF));

  /* perform a floorf */
  tmp = vcvt_f16_s16(vcvt_s16_f16(fx));

  /* if greater, substract 1 */
  uint16x4_t mask = vcgt_f16(tmp, fx);
  mask = vand_u16(mask, (uint16x4_t)(one));

  fx = vsub_f16(tmp, (float16x4_t)(mask));

  tmp = vmul_f16(fx, vdup_n_f16(c_cephes_exp_C1));
  float16x4_t z = vmul_f16(fx, vdup_n_f16(c_cephes_exp_C2));
  x = vsub_f16(x, tmp);
  x = vsub_f16(x, z);

  z = vmul_f16(x, x);

  float16x4_t y = vdup_n_f16(c_cephes_exp_p0);
  y = vfma_f16(vdup_n_f16(c_cephes_exp_p1), y, x);
  y = vfma_f16(vdup_n_f16(c_cephes_exp_p2), y, x);
  y = vfma_f16(vdup_n_f16(c_cephes_exp_p3), y, x);
  y = vfma_f16(vdup_n_f16(c_cephes_exp_p4), y, x);
  y = vfma_f16(vdup_n_f16(c_cephes_exp_p5), y, x);

  y = vfma_f16(x, y, z);
  y = vadd_f16(y, one);

  /* build 2^n */
  int16x4_t mm;
  mm = vcvt_s16_f16(fx);
  mm = vadd_s16(mm, vdup_n_s16(0xf));
  mm = vshl_n_s16(mm, 10);
  float16x4_t pow2n = vreinterpret_f16_s16(mm);

  y = vmul_f16(y, pow2n);
  return y;
}

static inline float16x8_t exp_ps_naive(float16x8_t x) {
  float16x8_t tmp, fx;

  float16x8_t one = vdupq_n_f16(1);
  x = vminq_f16(x, vdupq_n_f16(c_exp_hi_f16));
  x = vmaxq_f16(x, vdupq_n_f16(c_exp_lo_f16));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = vfmaq_f16(vdupq_n_f16(0.5f), x, vdupq_n_f16(c_cephes_LOG2EF));

  /* perform a floorf */
  tmp = vcvtq_f16_s16(vcvtq_s16_f16(fx));

  /* if greater, substract 1 */
  uint16x8_t mask = vcgtq_f16(tmp, fx);
  mask = vandq_u16(mask, vreinterpretq_u16_f16(one));

  fx = vsubq_f16(tmp, vreinterpretq_f16_u16(mask));

  tmp = vmulq_f16(fx, vdupq_n_f16(c_cephes_exp_C1));
  float16x8_t z = vmulq_f16(fx, vdupq_n_f16(c_cephes_exp_C2));
  x = vsubq_f16(x, tmp);
  x = vsubq_f16(x, z);

  z = vmulq_f16(x, x);

  float16x8_t y = vdupq_n_f16(c_cephes_exp_p0);
  y = vfmaq_f16(vdupq_n_f16(c_cephes_exp_p1), y, x);
  y = vfmaq_f16(vdupq_n_f16(c_cephes_exp_p2), y, x);
  y = vfmaq_f16(vdupq_n_f16(c_cephes_exp_p3), y, x);
  y = vfmaq_f16(vdupq_n_f16(c_cephes_exp_p4), y, x);
  y = vfmaq_f16(vdupq_n_f16(c_cephes_exp_p5), y, x);

  y = vfmaq_f16(x, y, z);
  y = vaddq_f16(y, one);

  /* build 2^n */
  int16x8_t mm;
  mm = vcvtq_s16_f16(fx);
  mm = vaddq_s16(mm, vdupq_n_s16(0xf));
  mm = vshlq_n_s16(mm, 10);
  float16x8_t pow2n = vreinterpretq_f16_s16(mm);

  y = vmulq_f16(y, pow2n);
  return y;
}

// exp() computed for 8 float at once
inline float16x8_t expq_ps_f16(float16x8_t x) {
  float32x4_t va = vcvt_f32_f16(vget_low_f16(x));
  float32x4_t vb = vcvt_f32_f16(vget_high_f16(x));
  float32x4_t vexpa = exp_ps(va);
  float32x4_t vexpb = exp_ps(vb);
  return vcombine_f16(vcvt_f16_f32(vexpa), vcvt_f16_f32(vexpb));
}

inline float16x4_t exp_ps_f16(float16x4_t x) {
  float32x4_t va = vcvt_f32_f16(x);
  float32x4_t vexpa = exp_ps(va);
  float16x4_t vresa = vcvt_f16_f32(vexpa);
  return vresa;
}

// exp_log = expq_ps_f16(vmulq_f16(b, log_ps(a)))
inline float16x8_t exp_logq_f16(float16x8_t a, float32x4_t b) {
  float32x4_t vsum_a_low = log_ps(vcvt_f32_f16(vget_low_f16(a)));
  float32x4_t vsum_a_high = log_ps(vcvt_f32_f16(vget_high_f16(a)));
  float32x4_t vsum_a = vmulq_f32(b, vsum_a_low);
  float32x4_t vsum_b = vmulq_f32(b, vsum_a_high);
  float32x4_t vres_a = exp_ps(vsum_a);
  float32x4_t vres_b = exp_ps(vsum_b);
  return vcombine_f16(vcvt_f16_f32(vres_b), vcvt_f16_f32(vres_a));
}

inline float16x4_t exp_log_f16(float16x4_t a, float32x4_t b) {
  float32x4_t vsum_a_low = log_ps(vcvt_f32_f16(a));
  float32x4_t vsum_a = vmulq_f32(b, vsum_a_low);
  float32x4_t vres_a = exp_ps(vsum_a);
  return vcvt_f16_f32(vres_a);
}

// pow(x, m) = exp(m * log(x))
inline float16x8_t powq_ps_f16(float16x8_t a, float32x4_t b) {
  float16x8_t vone = vdupq_n_f16(1.f);
  // x < 0
  for (int i = 0; i < 8; i++) {
    if (a[i] < 0) {
      a[i] = -a[i];
      if (static_cast<int>(b[i % 4]) % 2) {
        vone[i] = -1.f;
      }
    }
  }
  float16x8_t vsum = exp_logq_f16(a, b);
  return vmulq_f16(vsum, vone);
}

inline float16x4_t pow_ps_f16(float16x4_t a, float32x4_t b) {
  float16x4_t vone = vdup_n_f16(1.f);
  // x < 0
  for (int i = 0; i < 4; i++) {
    if (a[i] < 0) {
      a[i] = -a[i];
      if (static_cast<int>(b[i]) % 2) {
        vone[i] = -1.f;
      }
    }
  }
  float16x4_t vsum = exp_log_f16(a, b);
  return vmul_f16(vsum, vone);
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
