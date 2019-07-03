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

#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include "common/enforce.h"
#include "common/types.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#include "operators/math/math.h"
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0

inline ActivationType GetActivationType(const std::string &type) {
  if (type == "sigmoid") {
    return ActivationType::SIGMOID;
  } else if (type == "relu") {
    return ActivationType::RELU;
  } else if (type == "tanh") {
    return ActivationType::TANH;
  } else if (type == "identity" || type == "") {
    return ActivationType::IDENTITY;
  }
  PADDLE_MOBILE_THROW_EXCEPTION("Not support activation type.");
}

inline ActivationType GetActivationType(const int type) {
  if (type == 0) {
    return ActivationType::IDENTITY;
  } else if (type == 1) {
    return ActivationType::SIGMOID;
  } else if (type == 2) {
    return ActivationType::TANH;
  } else if (type == 3) {
    return ActivationType::RELU;
  }
  PADDLE_MOBILE_THROW_EXCEPTION("Not support activation type.");
}

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
template <ActivationType Act = IDENTITY>
inline float32x4_t vActiveq_f32(const float32x4_t &x) {
  return x;
}

template <>
inline float32x4_t vActiveq_f32<RELU>(const float32x4_t &x) {
  float32x4_t __zero = vdupq_n_f32(0.f);
  return vmaxq_f32(x, __zero);
}

template <>
inline float32x4_t vActiveq_f32<RELU6>(const float32x4_t &x) {
  float32x4_t __zero = vdupq_n_f32(0.f);
  float32x4_t __six = vdupq_n_f32(6.f);
  return vminq_f32(vmaxq_f32(x, __zero), __six);
}

template <>
inline float32x4_t vActiveq_f32<SIGMOID>(const float32x4_t &x) {
  float32x4_t __one = vdupq_n_f32(1.f);
  float32x4_t __x = vnegq_f32(x);
  __x = exp_ps(__x);
  __x = vaddq_f32(__x, __one);
  float32x4_t __out = vrecpeq_f32(__x);
  return vmulq_f32(vrecpsq_f32(__x, __out), __out);
}

template <>
inline float32x4_t vActiveq_f32<TANH>(const float32x4_t &x) {
  float32x4_t __one = vdupq_n_f32(1.f);
  float32x4_t __x = vnegq_f32(x);
  __x = vmulq_n_f32(__x, 2.f);
  __x = exp_ps(__x);
  __x = vaddq_f32(__x, __one);
  float32x4_t __out = vrecpeq_f32(__x);
  __out = vmulq_f32(vrecpsq_f32(__x, __out), __out);
  __out = vmulq_n_f32(__out, 2.f);
  return vsubq_f32(__out, __one);
}

template <>
inline float32x4_t vActiveq_f32<LOG>(const float32x4_t &x) {
  return log_ps(x);
}

template <ActivationType Act = IDENTITY>
inline float32x4_t vActiveq_f32(const float32x4_t &x,
                                const float32x4_t &alpha) {
  return x;
}

template <>
inline float32x4_t vActiveq_f32<LEAKY_RELU>(const float32x4_t &x,
                                            const float32x4_t &alpha) {
  return vmaxq_f32(x, vmulq_f32(x, alpha));
}

template <>
inline float32x4_t vActiveq_f32<RELU6>(const float32x4_t &x,
                                       const float32x4_t &alpha) {
  float32x4_t __zero = vdupq_n_f32(0.f);
  float32x4_t __threshold = vdupq_n_f32(vgetq_lane_f32(alpha, 0));
  return vminq_f32(vmaxq_f32(x, __zero), __threshold);
}
#endif

template <ActivationType Act = IDENTITY>
inline float Active(const float &x) {
  return x;
}

template <>
inline float Active<RELU>(const float &x) {
  return std::max(x, 0.f);
}

template <>
inline float Active<RELU6>(const float &x) {
  return std::min(std::max(x, 0.f), 6.f);
}

template <>
inline float Active<SIGMOID>(const float &x) {
  //  float tmp = x > SIGMOID_THRESHOLD_MAX ? SIGMOID_THRESHOLD_MAX : x;
  //  tmp = x > SIGMOID_THRESHOLD_MIN ? x : SIGMOID_THRESHOLD_MIN;
  //  return 1.f / (1.f + exp(-tmp));
  return 1.f / (1.f + exp(-x));
}

template <>
inline float Active<TANH>(const float &x) {
  //  float tmp = -2.f * x;
  //  tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
  //  return (2.f / (1.f + exp(tmp))) - 1.f;
  return 2.f / (1.f + exp(-2.f * x)) - 1.f;
}

template <>
inline float Active<LOG>(const float &x) {
  return log(x);
}

template <ActivationType Act = IDENTITY>
inline float Active(const float &x, const float &alpha) {
  return x;
}

template <>
inline float Active<LEAKY_RELU>(const float &x, const float &alpha) {
  return std::max(x, alpha * x);
}

template <>
inline float Active<RELU6>(const float &x, const float &alpha) {
  return std::min(std::max(x, 0.f), alpha);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
