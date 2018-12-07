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
#include "common/types.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
template <ActivationType Act = Linear>
inline float32x4_t vActiveq_f32(const float32x4_t &x) {
  return x;
}

template <>
inline float32x4_t vActiveq_f32<Relu>(const float32x4_t &x) {
  float32x4_t __zero = vdupq_n_f32(0.f);
  return vmaxq_f32(x, __zero);
}

template <>
inline float32x4_t vActiveq_f32<Relu6>(const float32x4_t &x) {
  float32x4_t __zero = vdupq_n_f32(0.f);
  float32x4_t __six = vdupq_n_f32(6.f);
  return vminq_f32(vmaxq_f32(x, __zero), __six);
}
#endif

template <ActivationType Act = Linear>
inline float Active(const float &x) {
  return x;
}

template <>
inline float Active<Relu>(const float &x) {
  return std::max(x, 0.f);
}

template <>
inline float Active<Relu6>(const float &x) {
  return std::min(std::max(x, 0.f), 6.f);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
