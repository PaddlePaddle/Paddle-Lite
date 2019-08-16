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

#ifdef POOL_OP

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "common/types.h"
#include "framework/tensor.h"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

template <PoolingType P = MAX>
struct PoolingVal {
  float val;
  int count;
  PoolingVal() : count(0) { val = -std::numeric_limits<float>::max(); }
  inline PoolingVal<P> &operator+=(const float &x) {
    val = std::max(val, x);
    ++count;
    return *this;
  }
  inline float Value() { return (count > 0) ? val : 0.f; }
  inline float ExclusiveSum(int total) {
    return ((count > 0) ? val : 0.f) * total;
  }
};

template <>
struct PoolingVal<AVG> {
  float val;
  int count;
  PoolingVal() : val(0.f), count(0) {}
  inline PoolingVal<AVG> &operator+=(const float &x) {
    val += x;
    ++count;
    return *this;
  }
  inline float Value() { return (count > 0) ? val * (1.f / count) : 0.f; }
  inline float ExclusiveSum(int total) { return (count > 0) ? val : 0.f; }
};

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
template <PoolingType P = MAX>
inline float32x4_t vPoolInitq_f32() {
  return vdupq_n_f32(-std::numeric_limits<float>::max());
}

template <>
inline float32x4_t vPoolInitq_f32<AVG>() {
  return vdupq_n_f32(0.f);
}

template <PoolingType P = MAX>
inline float32x2_t vPoolInit_f32() {
  return vdup_n_f32(-std::numeric_limits<float>::max());
}

template <>
inline float32x2_t vPoolInit_f32<AVG>() {
  return vdup_n_f32(0.f);
}

template <PoolingType P = MAX>
inline float32x4_t vPoolPreq_f32(const float32x4_t &x1, const float32x4_t &x2) {
  return vmaxq_f32(x1, x2);
}

template <>
inline float32x4_t vPoolPreq_f32<AVG>(const float32x4_t &x1,
                                      const float32x4_t &x2) {
  return vaddq_f32(x1, x2);
}

template <PoolingType P = MAX>
inline float32x2_t vPoolPre_f32(const float32x2_t &x1, const float32x2_t &x2) {
  return vmax_f32(x1, x2);
}

template <>
inline float32x2_t vPoolPre_f32<AVG>(const float32x2_t &x1,
                                     const float32x2_t &x2) {
  return vadd_f32(x1, x2);
}

template <PoolingType P = MAX>
inline float32x2_t vpPoolPre_f32(const float32x2_t &x1, const float32x2_t &x2) {
  return vpmax_f32(x1, x2);
}

template <>
inline float32x2_t vpPoolPre_f32<AVG>(const float32x2_t &x1,
                                      const float32x2_t &x2) {
  return vpadd_f32(x1, x2);
}

template <PoolingType P = MAX>
inline float32x4_t vPoolPostq_f32(const float32x4_t &x,
                                  const float32x4_t &post) {
  return x;
}

template <>
inline float32x4_t vPoolPostq_f32<AVG>(const float32x4_t &x,
                                       const float32x4_t &post) {
  return vmulq_f32(x, post);
}

template <PoolingType P = MAX>
inline float32x2_t vPoolPost_f32(const float32x2_t &x,
                                 const float32x2_t &post) {
  return x;
}

template <>
inline float32x2_t vPoolPost_f32<AVG>(const float32x2_t &x,
                                      const float32x2_t &post) {
  return vmul_f32(x, post);
}
#endif  // __ARM_NEON__

template <PoolingType P = MAX>
inline float PoolPre(const float &x1, const float &x2) {
  return std::max(x1, x2);
}

template <>
inline float PoolPre<AVG>(const float &x1, const float &x2) {
  return x1 + x2;
}

template <PoolingType P = MAX>
inline float PoolPost(const float &x, const float &post) {
  return x;
}

template <>
inline float PoolPost<AVG>(const float &x, const float &post) {
  return x * post;
}

template <PoolingType P>
struct Pooling {
  void operator()(const framework::Tensor &input,
                  const std::vector<int> &kernel_size,
                  const std::vector<int> &strides,
                  const std::vector<int> &paddings, framework::Tensor *output);
};

template <PoolingType P, int Stride>
struct Pooling2x2 {
  void operator()(const framework::Tensor &input,
                  const std::vector<int> &paddings, framework::Tensor *output);
};

template <PoolingType P, int Stride>
struct Pooling3x3 {
  void operator()(const framework::Tensor &input,
                  const std::vector<int> &paddings, const bool exclusive,
                  framework::Tensor *output);
};

template <PoolingType P, int Stride>
struct Pooling5x5 {
  void operator()(const framework::Tensor &input,
                  const std::vector<int> &paddings, framework::Tensor *output);
};

template <PoolingType P, int Stride>
struct Pooling7x7 {
  void operator()(const framework::Tensor &input,
                  const std::vector<int> &paddings, framework::Tensor *output);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
