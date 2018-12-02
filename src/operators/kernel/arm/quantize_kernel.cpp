/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef QUANT_OP

#include "operators/kernel/quantize_kernel.h"
#include <cmath>

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>

namespace paddle_mobile {
namespace operators {

#ifndef __aarch64__
inline float32_t vmaxvq_f32(float32x4_t r) {
  float32x2_t v = vmax_f32(vget_high_f32(r), vget_low_f32(r));
  return vget_lane_f32(vpmax_f32(v, v), 0);
}
#endif

template <RoundType R = ROUND_NEAREST_TOWARDS_ZERO>
inline int32x4_t vround_f32(float32x4_t r) {
  return vcvtq_s32_f32(r);
}

template <>
inline int32x4_t vround_f32<ROUND_NEAREST_AWAY_ZERO>(float32x4_t r) {
  float32x4_t plus = vdupq_n_f32(0.5);
  float32x4_t minus = vdupq_n_f32(-0.5);
  float32x4_t zero = vdupq_n_f32(0);
  uint32x4_t more_than_zero = vcgtq_f32(r, zero);
  float32x4_t temp = vbslq_f32(more_than_zero, plus, minus);
  temp = vaddq_f32(r, temp);
  int32x4_t ret = vcvtq_s32_f32(temp);
  return ret;
}

template <>
inline int32x4_t vround_f32<ROUND_NEAREST_TO_EVEN>(float32x4_t r) {
  float32x4_t point5 = vdupq_n_f32(0.5);
  int32x4_t one = vdupq_n_s32(1);
  int32x4_t zero = vdupq_n_s32(0);

  int32x4_t rnd = vround_f32<ROUND_NEAREST_AWAY_ZERO>(r);
  float32x4_t frnd = vcvtq_f32_s32(rnd);
  frnd = vsubq_f32(frnd, r);
  frnd = vabsq_f32(frnd);
  uint32x4_t equal_point5 = vceqq_f32(frnd, point5);
  int32x4_t abs_rnd = vabsq_s32(rnd);
  abs_rnd = vandq_s32(abs_rnd, one);
  uint32x4_t not_mod2 = vreinterpretq_u32_s32(abs_rnd);
  uint32x4_t mask = vandq_u32(equal_point5, not_mod2);
  uint32x4_t more_than_zero = vcgtq_s32(rnd, zero);
  more_than_zero = vandq_u32(more_than_zero, vreinterpretq_u32_s32(one));
  mask = veorq_u32(more_than_zero, mask);
  more_than_zero = veorq_u32(more_than_zero, vreinterpretq_u32_s32(one));
  mask = vaddq_u32(more_than_zero, mask);
  int32x4_t smask = vreinterpretq_s32_u32(mask);
  smask = vsubq_s32(smask, one);
  rnd = vaddq_s32(rnd, smask);
  return rnd;
}
#endif

template <RoundType R = ROUND_NEAREST_TOWARDS_ZERO>
inline int8_t Round(const float &x) {
  return static_cast<int8_t>(x);
}

template <>
inline int8_t Round<ROUND_NEAREST_AWAY_ZERO>(const float &x) {
  return std::round(x);
}

template <>
inline int8_t Round<ROUND_NEAREST_TO_EVEN>(const float &x) {
  float v = std::round(x);
  int32_t q = static_cast<int32_t>(v);
  if (std::abs(std::abs(q - v) - 0.5) <= 0) {
    if (std::abs(q) % 2 != 0) {
      q = q + ((q > 0) ? -1 : 1);
    }
  }
  return static_cast<int8_t>(q);
}

template <RoundType R>
static void Quantize(const Tensor *input, const float scale, Tensor *output) {
  const float *x = input->data<const float>();
  int8_t *y = output->mutable_data<int8_t>();
  size_t remain = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  size_t loop = remain >> 4;
  remain = remain & 0xF;

  #pragma omp parallel for
  for (size_t i = 0; i < loop; ++i) {
    const float *local_x = x + (i << 4);
    int8_t *local_y = y + (i << 4);
    float32x4_t r0 = vld1q_f32(local_x);
    float32x4_t r1 = vld1q_f32(local_x + 4);
    float32x4_t r2 = vld1q_f32(local_x + 8);
    float32x4_t r3 = vld1q_f32(local_x + 12);
    r0 = vmulq_n_f32(r0, scale);
    r1 = vmulq_n_f32(r1, scale);
    r2 = vmulq_n_f32(r2, scale);
    r3 = vmulq_n_f32(r3, scale);
    int32x4_t q0 = vround_f32<R>(r0);
    int32x4_t q1 = vround_f32<R>(r1);
    int32x4_t q2 = vround_f32<R>(r2);
    int32x4_t q3 = vround_f32<R>(r3);
    int16x4_t d0 = vmovn_s32(q0);
    int16x4_t d1 = vmovn_s32(q1);
    int16x4_t d2 = vmovn_s32(q2);
    int16x4_t d3 = vmovn_s32(q3);
    int16x8_t q5 = vcombine_s16(d0, d1);
    int16x8_t q6 = vcombine_s16(d2, d3);
    int8x8_t d5 = vmovn_s16(q5);
    int8x8_t d6 = vmovn_s16(q6);
    vst1_s8(local_y, d5);
    vst1_s8(local_y + 8, d6);
  }
  x += (loop << 4);
  y += (loop << 4);
#endif
  for (size_t i = 0; i < remain; ++i) {
    y[i] = Round<R>(x[i] * scale);
  }
}

float find_abs_max(const Tensor *input) {
  float max_abs = 0.f;
  const float *x = input->data<const float>();
  size_t remain = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  size_t loop = remain >> 4;
  remain = remain & 0xF;
  float32x4_t __max = {0.f, 0.f, 0.f, 0.f};

  for (size_t i = 0; i < loop; ++i, x += 16) {
    float32x4_t r0 = vld1q_f32(x);
    float32x4_t r1 = vld1q_f32(x + 4);
    float32x4_t r2 = vld1q_f32(x + 8);
    float32x4_t r3 = vld1q_f32(x + 12);
    r0 = vabsq_f32(r0);
    r1 = vabsq_f32(r1);
    r2 = vabsq_f32(r2);
    r3 = vabsq_f32(r3);
    r0 = vmaxq_f32(r0, r1);
    r1 = vmaxq_f32(r2, r3);
    r0 = vmaxq_f32(r0, r1);
    __max = vmaxq_f32(r0, __max);
  }
  max_abs = vmaxvq_f32(__max);
#endif
  for (size_t i = 0; i < remain; ++i) {
    max_abs = std::max(max_abs, std::abs(x[i]));
  }
  return max_abs;
}

template <>
bool QuantizeKernel<CPU, float>::Init(QuantizeParam<CPU> *param) {
  return true;
}

template <>
void QuantizeKernel<CPU, float>::Compute(const QuantizeParam<CPU> &param) {
  const Tensor *input = param.input_;
  Tensor *output = param.output_;
  Tensor *output_scale = param.online_scale_;
  float max_abs = 0.f;
  if (param.is_static_) {
    max_abs = param.static_scale_;
  } else {
    max_abs = find_abs_max(input);
  }
  max_abs = std::max(max_abs, 1e-6f);
  // only support int8 currently
  float scale = 127 / max_abs;
  param.online_scale_->mutable_data<float>()[0] = max_abs;
  switch (param.round_type_) {
    case ROUND_NEAREST_TO_EVEN:
      Quantize<ROUND_NEAREST_TO_EVEN>(input, scale, output);
      break;
    case ROUND_NEAREST_TOWARDS_ZERO:
      Quantize<ROUND_NEAREST_TOWARDS_ZERO>(input, scale, output);
      break;
    case ROUND_NEAREST_AWAY_ZERO:
      Quantize<ROUND_NEAREST_AWAY_ZERO>(input, scale, output);
      break;
    default:
      LOG(kLOG_ERROR) << "round type is not supported.";
      break;
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
