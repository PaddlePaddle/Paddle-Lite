// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <arm_neon.h>

#include "lite/backends/arm/math/elementwise_naive_impl.h"

#pragma once
/**
 * These macros will convert clang marco into function call
 */
#ifdef __clang__
#define __ai static inline __attribute__((__always_inline__, __nodebug__))

__ai int32x4_t vld1q_s32_wrap(const int32_t* p0) { return vld1q_s32(p0); }
#undef vld1q_s32
#define vld1q_s32 vld1q_s32_wrap

__ai void vst1q_s32_wrap(int32_t* a, int32x4_t b) { return vst1q_s32(a, b); }
#undef vst1q_s32
#define vst1q_s32 vst1q_s32_wrap

__ai float32x4_t vld1q_f32_wrap(const float* p0) { return vld1q_f32(p0); }
#undef vld1q_f32
#define vld1q_f32 vld1q_f32_wrap

__ai void vst1q_f32_wrap(float* a, float32x4_t b) { return vst1q_f32(a, b); }
#undef vst1q_f32
#define vst1q_f32 vst1q_f32_wrap

#undef __ai
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {

static inline float32x4_t __attribute__((__always_inline__))
neon_relu_float(const float32x4_t& a) {
  constexpr float32x4_t zero = {0, 0, 0, 0};
  return vmaxq_f32(a, zero);
}

struct NullNeonConfig {};

template <class Config1, class Config2>
struct MergeConfig : public Config1, public Config2 {};

/**
 * see neon_elementwise_range_to_one to get how to use Config
 */
template <class DataType>
struct BasicConfig;

template <>
struct BasicConfig<int32_t> {
  using T = int32_t;
  using NeonT = int32x4_t;
  constexpr static auto neon_dup = vdupq_n_s32;
  constexpr static auto neon_ld = vld1q_s32;
  constexpr static auto neon_st = vst1q_s32;
};

template <>
struct BasicConfig<float> {
  using T = float;
  using NeonT = float32x4_t;
  constexpr static auto neon_dup = vdupq_n_f32;
  constexpr static auto neon_ld = vld1q_f32;
  constexpr static auto neon_st = vst1q_f32;
};

enum class ActiveType { NO_ACTIVE, RELU };

template <ActiveType, class DataType>
struct ActiveConfig {};

template <class DataType>
struct ActiveConfig<ActiveType::NO_ACTIVE, DataType> {
  constexpr static DataType (*naive_active)(DataType) = nullptr;
  constexpr static typename BasicConfig<DataType>::NeonT (*neon_active)(
      const typename BasicConfig<DataType>::NeonT&) = nullptr;
};

template <>
struct ActiveConfig<ActiveType::RELU, float> {
  constexpr static float (*naive_active)(float) = naive_relu<float>;
  constexpr static float32x4_t (*neon_active)(const float32x4_t&) =
      neon_relu_float;
};

template <class T>
struct AddConfig {};

template <>
struct AddConfig<int32_t> : public BasicConfig<int32_t> {
  constexpr static auto naive_op = naive_add<int32_t>;
  constexpr static auto neon_op = vaddq_s32;
};
template <>
struct AddConfig<float> : public BasicConfig<float> {
  constexpr static auto naive_op = naive_add<float>;
  constexpr static auto neon_op = vaddq_f32;
};

template <class T>
struct SubConfig {};

template <>
struct SubConfig<int32_t> : public BasicConfig<int32_t> {
  constexpr static auto naive_op = naive_sub<int32_t>;
  constexpr static auto neon_op = vsubq_s32;
};

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
