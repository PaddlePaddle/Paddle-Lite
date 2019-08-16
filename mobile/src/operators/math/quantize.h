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

#pragma once

#include <cmath>
#include "common/types.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

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
  if (fabs(fabs(q - v) - 0.5) <= 0) {
    if (abs(q) % 2 != 0) {
      q = q + ((q > 0) ? -1 : 1);
    }
  }
  return static_cast<int8_t>(q);
}

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
template <RoundType R = ROUND_NEAREST_TOWARDS_ZERO>
inline int32x4_t vRoundq_f32(const float32x4_t &x) {
  return vcvtq_s32_f32(x);
}

template <>
inline int32x4_t vRoundq_f32<ROUND_NEAREST_AWAY_ZERO>(const float32x4_t &x) {
#if __aarch64__
  return vcvtaq_s32_f32(x);
#else
  float32x4_t plus = vdupq_n_f32(0.5);
  float32x4_t minus = vdupq_n_f32(-0.5);
  float32x4_t zero = vdupq_n_f32(0);
  uint32x4_t more_than_zero = vcgtq_f32(x, zero);
  float32x4_t temp = vbslq_f32(more_than_zero, plus, minus);
  temp = vaddq_f32(x, temp);
  int32x4_t ret = vcvtq_s32_f32(temp);
  return ret;
#endif
}

template <>
inline int32x4_t vRoundq_f32<ROUND_NEAREST_TO_EVEN>(const float32x4_t &x) {
#if __aarch64__
  return vcvtnq_s32_f32(x);
#else
  float32x4_t point5 = vdupq_n_f32(0.5);
  int32x4_t one = vdupq_n_s32(1);
  int32x4_t zero = vdupq_n_s32(0);

  int32x4_t rnd = math::vRoundq_f32<ROUND_NEAREST_AWAY_ZERO>(x);
  float32x4_t frnd = vcvtq_f32_s32(rnd);
  frnd = vsubq_f32(frnd, x);
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
#endif
}
#endif  // __ARM_NEON__

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // QUANT_OP
