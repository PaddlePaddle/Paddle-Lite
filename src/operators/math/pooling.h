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

template <PoolingType P = Max>
struct PoolingVal {
  float val;
  int count;
  PoolingVal() {
    val = -std::numeric_limits<float>::max();
    count = 0;
  }
  inline PoolingVal<P> &operator+=(const float &x) {
    val = std::max(val, x);
    count += 1;
    return *this;
  }
  float Value() const {
    if (count > 0) {
      return val;
    }
    return 0.f;
  }
};

template <>
struct PoolingVal<Avg> {
  float val;
  int count;
  PoolingVal() {
    val = 0.f;
    count = 0;
  }
  inline PoolingVal<Avg> &operator+=(const float &x) {
    val += x;
    count += 1;
    return *this;
  }
  float Value() const {
    if (count > 0) {
      return val / count;
    }
    return 0.f;
  }
};

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
template <PoolingType P = Max>
inline float32x4_t vPoolPreq_f32(const float32x4_t &x1, const float32x4_t &x2) {
  return vmaxq_f32(x1, x2);
}

template <>
inline float32x4_t vPoolPreq_f32<Avg>(const float32x4_t &x1,
                                      const float32x4_t &x2) {
  return vaddq_f32(x1, x2);
}

template <PoolingType P = Max>
inline float32x4_t vPoolPostq_f32(const float32x4_t &x) {
  return x;
}

template <>
inline float32x4_t vPoolPostq_f32<Avg>(const float32x4_t &x) {
  float32x4_t avg = vdupq_n_f32(1.f / 9);
  return vmulq_f32(avg, x);
}
#endif  // __ARM_NEON__

template <PoolingType P = Max>
inline float PoolPre(const float &x1, const float &x2) {
  return std::max(x1, x2);
}

template <>
inline float PoolPre<Avg>(const float &x1, const float &x2) {
  return x1 + x2;
}

template <PoolingType P = Max>
inline float PoolPost(const float &x) {
  return x;
}

template <>
inline float PoolPost<Avg>(const float &x) {
  return 1.f / 9 * x;
}

template <PoolingType P>
struct Pooling {
  inline void operator()(const framework::Tensor &input,
                         const std::vector<int> &kernel_size,
                         const std::vector<int> &strides,
                         const std::vector<int> &paddings,
                         framework::Tensor *output);
};

template <PoolingType P, int Stride>
struct Pooling2x2 {
  inline void operator()(const framework::Tensor &input,
                         const std::vector<int> &paddings,
                         framework::Tensor *output);
};

template <PoolingType P, int Stride>
struct Pooling3x3 {
  inline void operator()(const framework::Tensor &input,
                         const std::vector<int> &paddings,
                         framework::Tensor *output);
};

template <PoolingType P, int Stride>
struct Pooling5x5 {
  inline void operator()(const framework::Tensor &input,
                         const std::vector<int> &paddings,
                         framework::Tensor *output);
};

template <PoolingType P, int Stride>
struct Pooling7x7 {
  inline void operator()(const framework::Tensor &input,
                         const std::vector<int> &paddings,
                         framework::Tensor *output);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
