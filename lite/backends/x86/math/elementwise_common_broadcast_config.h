// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>
#include "lite/backends/x86/math/activation_functions.h"

#if defined(__AVX__) && !defined(__SSE4_2__)
#define __SSE4_2__ 1
#endif

#if defined(__AVX__)
#include <immintrin.h>
#endif
#if defined(__SSE4_2__)
#include <emmintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#endif

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

//****************************** isa op *******************************
template <typename OUT_T, typename IN_T>
OUT_T loadu_ps_inline(const IN_T* a);

template <typename OUT_T, typename IN_T>
void storeu_ps_inline(IN_T* b, OUT_T a);

template <typename OUT_T, typename IN_T>
OUT_T set1_ps_inline(IN_T a);

template <typename T>
T add_ps_inline(T a, T b);

template <typename T>
T sub_ps_inline(T a, T b);

template <typename T>
T max_ps_inline(T a, T b);

template <typename T>
T min_ps_inline(T a, T b);

template <typename T>
T div_ps_inline(T a, T b);

template <typename T>
T mul_ps_inline(T a, T b);

template <typename OUT_T, typename IN_T>
OUT_T set1_epi32_inline(IN_T a);

template <typename OUT_T, typename IN_T>
OUT_T set1_epi64x_inline(IN_T a);

template <typename OUT_T, typename IN_T>
OUT_T loadu_si_inline(const IN_T* a);

template <typename OUT_T, typename IN_T>
void storeu_si_inline(IN_T* b, OUT_T a);

template <typename T>
T add_epi32_inline(T a, T b);

template <typename T>
T add_epi64_inline(T a, T b);

template <typename T>
T sub_epi32_inline(T a, T b);

template <typename T>
T sub_epi64_inline(T a, T b);

template <typename T>
T mul_epi32_inline(T a, T b);

template <typename T>
T max_epi32_inline(T a, T b);

template <typename T>
T min_epi32_inline(T a, T b);

// compiler can't recognize intrinsics function name
#ifdef __AVX__
template <>
__m256 loadu_ps_inline<__m256, float>(const float* a) {
  return _mm256_loadu_ps(a);
}
template <>
void storeu_ps_inline<__m256, float>(float* b, __m256 a) {
  _mm256_storeu_ps(b, a);
}
template <>
__m256 set1_ps_inline<__m256, float>(float a) {
  return _mm256_set1_ps(a);
}
template <>
__m256 add_ps_inline<__m256>(__m256 a, __m256 b) {
  return _mm256_add_ps(a, b);
}
template <>
__m256 sub_ps_inline<__m256>(__m256 a, __m256 b) {
  return _mm256_sub_ps(a, b);
}
template <>
__m256 max_ps_inline<__m256>(__m256 a, __m256 b) {
  return _mm256_max_ps(a, b);
}
template <>
__m256 min_ps_inline<__m256>(__m256 a, __m256 b) {
  return _mm256_min_ps(a, b);
}
template <>
__m256 div_ps_inline<__m256>(__m256 a, __m256 b) {
  return _mm256_div_ps(a, b);
}
template <>
__m256 mul_ps_inline<__m256>(__m256 a, __m256 b) {
  return _mm256_mul_ps(a, b);
}
#elif defined(__SSE4_2__)
template <>
__m128 loadu_ps_inline<__m128, float>(const float* a) {
  return _mm_loadu_ps(a);
}
template <>
void storeu_ps_inline<__m128, float>(float* b, __m128 a) {
  _mm_storeu_ps(b, a);
}
template <>
__m128 set1_ps_inline<__m128, float>(float a) {
  return _mm_set1_ps(a);
}
template <>
__m128 add_ps_inline<__m128>(__m128 a, __m128 b) {
  return _mm_add_ps(a, b);
}
template <>
__m128 sub_ps_inline<__m128>(__m128 a, __m128 b) {
  return _mm_sub_ps(a, b);
}
template <>
__m128 max_ps_inline<__m128>(__m128 a, __m128 b) {
  return _mm_max_ps(a, b);
}
template <>
__m128 min_ps_inline<__m128>(__m128 a, __m128 b) {
  return _mm_min_ps(a, b);
}
template <>
__m128 div_ps_inline<__m128>(__m128 a, __m128 b) {
  return _mm_div_ps(a, b);
}
template <>
__m128 mul_ps_inline<__m128>(__m128 a, __m128 b) {
  return _mm_mul_ps(a, b);
}

__m128 _mm_relu_ps(const __m128& a) {
  __m128 vec_zero = _mm_set1_ps(0.f);
  return _mm_max_ps(a, vec_zero);
}
#endif

#if defined(__AVX2__)
template <>
__m256i loadu_si_inline<__m256i, __m256i>(const __m256i* a) {
  return _mm256_loadu_si256(a);
}
template <>
void storeu_si_inline<__m256i, __m256i>(__m256i* b, __m256i a) {
  _mm256_storeu_si256(b, a);
}
template <>
__m256i set1_epi32_inline<__m256i, int>(int a) {
  return _mm256_set1_epi32(a);
}
template <>
__m256i set1_epi64x_inline<__m256i, int64_t>(int64_t a) {
  return _mm256_set1_epi64x(a);
}
template <>
__m256i add_epi32_inline<__m256i>(__m256i a, __m256i b) {
  return _mm256_add_epi32(a, b);
}
template <>
__m256i add_epi64_inline<__m256i>(__m256i a, __m256i b) {
  return _mm256_add_epi64(a, b);
}
template <>
__m256i sub_epi32_inline<__m256i>(__m256i a, __m256i b) {
  return _mm256_sub_epi32(a, b);
}
template <>
__m256i sub_epi64_inline<__m256i>(__m256i a, __m256i b) {
  return _mm256_sub_epi64(a, b);
}
template <>
__m256i mul_epi32_inline<__m256i>(__m256i a, __m256i b) {
  return _mm256_mullo_epi32(a, b);
}
template <>
__m256i max_epi32_inline<__m256i>(__m256i a, __m256i b) {
  return _mm256_max_epi32(a, b);
}
template <>
__m256i min_epi32_inline<__m256i>(__m256i a, __m256i b) {
  return _mm256_min_epi32(a, b);
}
#elif defined(__SSE4_2__)
template <>
__m128i loadu_si_inline<__m128i, __m128i>(const __m128i* a) {
  return _mm_loadu_si128(a);
}
template <>
void storeu_si_inline<__m128i, __m128i>(__m128i* b, __m128i a) {
  _mm_storeu_si128(b, a);
}
template <>
__m128i set1_epi32_inline<__m128i, int>(int a) {
  return _mm_set1_epi32(a);
}
template <>
__m128i set1_epi64x_inline<__m128i, int64_t>(int64_t a) {
  return _mm_set1_epi64x(a);
}
template <>
__m128i add_epi32_inline<__m128i>(__m128i a, __m128i b) {
  return _mm_add_epi32(a, b);
}
template <>
__m128i add_epi64_inline<__m128i>(__m128i a, __m128i b) {
  return _mm_add_epi64(a, b);
}
template <>
__m128i sub_epi32_inline<__m128i>(__m128i a, __m128i b) {
  return _mm_sub_epi32(a, b);
}
template <>
__m128i sub_epi64_inline<__m128i>(__m128i a, __m128i b) {
  return _mm_sub_epi64(a, b);
}
template <>
__m128i mul_epi32_inline<__m128i>(__m128i a, __m128i b) {
  return _mm_mullo_epi32(a, b);
}
template <>
__m128i max_epi32_inline<__m128i>(__m128i a, __m128i b) {
  return _mm_max_epi32(a, b);
}
template <>
__m128i min_epi32_inline<__m128i>(__m128i a, __m128i b) {
  return _mm_min_epi32(a, b);
}
#endif

//****************************** naive op *******************************
template <class T>
inline T NaiveRelu(T a) {
  return a > 0 ? a : 0;
}

template <class T>
inline T NaiveTanh(T a) {
  float x = expf(a);
  float y = expf(-a);
  return (x - y) / (x + y);
}

template <class T>
inline T NaiveSigmoid(T a) {
  const T min = -40.0;  // SIGMOID_THRESHOLD_MIN;
  const T max = 13.0;   // SIGMOID_THRESHOLD_MAX;
  T tmp = (a < min) ? min : ((a > max) ? max : a);
  return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-tmp));
}

template <typename T>
inline T NaiveAdd(T l, T r) {
  return l + r;
}

template <typename T>
inline T NaiveSub(T l, T r) {
  return l - r;
}

template <typename T>
inline T NaiveMul(T l, T r) {
  return l * r;
}

template <typename T>
inline T NaiveDiv(T l, T r) {
  return l / r;
}

template <typename T>
inline T NaiveFloorDiv(T l, T r) {
  return static_cast<T>(std::trunc(l / r));
}

template <typename T>
inline T NaiveMax(T l, T r) {
  return l > r ? l : r;
}

template <typename T>
inline T NaiveMin(T l, T r) {
  return l < r ? l : r;
}

template <typename T>
inline T NaiveMod(T l, T r) {
  T res = l % r;
  if ((res != 0) && ((res < 0) != (r < 0))) res += r;
  return res;
}

template <typename T>
inline T NaivePow(T l, T r) {
  return std::pow(l, r);
}

//*************************** Config Struct *****************************
struct NullCpuInstruction {};

template <class ComputeConfig, class ActConfig>
struct MergeConfig : public ComputeConfig, public ActConfig {};

enum class ActiveType { NO_ACTIVE, RELU, TANH, SIGMOID };

template <class DataType>
struct BasicConfig {};

template <ActiveType, class DataType>
struct ActiveConfig {};

template <class T>
struct AddConfig {};

template <class T>
struct SubConfig {};

template <class T>
struct MulConfig {};

template <class T>
struct MaxConfig {};

template <class T>
struct MinConfig {};

template <class T>
struct DivConfig {};

template <class T>
struct FloorDivConfig {};

template <class T>
struct ModConfig {};

template <class T>
struct PowConfig {};

//***************************** float Config *********************
template <>
struct BasicConfig<float> {
#if defined(__AVX__)
  using T = float;
  using ISA_T = __m256;
  using LD_T = float;  // using for load and store
#elif defined(__SSE4_2__)
  using T = float;
  using ISA_T = __m128;
  using LD_T = float;
#endif
  constexpr static auto isa_dup = set1_ps_inline<ISA_T, T>;
  constexpr static auto isa_ld = loadu_ps_inline<ISA_T, LD_T>;
  constexpr static auto isa_str = storeu_ps_inline<ISA_T, LD_T>;
};

template <>
struct AddConfig<float> : public BasicConfig<float> {
  constexpr static auto naive_op = NaiveAdd<float>;
  constexpr static auto isa_op =
      add_ps_inline<typename BasicConfig<float>::ISA_T>;
};

template <>
struct SubConfig<float> : public BasicConfig<float> {
  constexpr static auto naive_op = NaiveSub<float>;
  constexpr static auto isa_op =
      sub_ps_inline<typename BasicConfig<float>::ISA_T>;
};

template <>
struct MulConfig<float> : public BasicConfig<float> {
  constexpr static auto naive_op = NaiveMul<float>;
  constexpr static auto isa_op =
      mul_ps_inline<typename BasicConfig<float>::ISA_T>;
};

template <>
struct MaxConfig<float> : public BasicConfig<float> {
  constexpr static auto naive_op = NaiveMax<float>;
  constexpr static auto isa_op =
      max_ps_inline<typename BasicConfig<float>::ISA_T>;
};

template <>
struct MinConfig<float> : public BasicConfig<float> {
  constexpr static auto naive_op = NaiveMin<float>;
  constexpr static auto isa_op =
      min_ps_inline<typename BasicConfig<float>::ISA_T>;
};

template <>
struct DivConfig<float> : public BasicConfig<float> {
  constexpr static auto naive_op = NaiveDiv<float>;
  constexpr static auto isa_op =
      div_ps_inline<typename BasicConfig<float>::ISA_T>;
};

//************************** in32,int64 ******************
template <>
struct BasicConfig<int32_t> {
#if defined(__AVX2__)
  using T = int32_t;
  using ISA_T = __m256i;
  using LD_T = __m256i;
#elif defined(__SSE4_2__)
  using T = int32_t;
  using ISA_T = __m128i;
  using LD_T = __m128i;
#endif
  constexpr static auto isa_dup = set1_epi32_inline<ISA_T, T>;
  constexpr static auto isa_ld = loadu_si_inline<ISA_T, LD_T>;
  constexpr static auto isa_str = storeu_si_inline<ISA_T, LD_T>;
};

template <>
struct BasicConfig<int64_t> {
#if defined(__AVX2__)
  using T = int64_t;
  using ISA_T = __m256i;
  using LD_T = __m256i;
#elif defined(__SSE4_2__)
  using T = int64_t;
  using ISA_T = __m128i;
  using LD_T = __m128i;
#endif
  constexpr static auto isa_dup = set1_epi64x_inline<ISA_T, T>;
  constexpr static auto isa_ld = loadu_si_inline<ISA_T, LD_T>;
  constexpr static auto isa_str = storeu_si_inline<ISA_T, LD_T>;
};

template <>
struct AddConfig<int32_t> : public BasicConfig<int32_t> {
  constexpr static auto naive_op = NaiveAdd<int32_t>;
  constexpr static auto isa_op =
      add_epi32_inline<typename BasicConfig<int32_t>::ISA_T>;
};

template <>
struct AddConfig<int64_t> : public BasicConfig<int64_t> {
  constexpr static auto naive_op = NaiveAdd<int64_t>;
  constexpr static auto isa_op =
      add_epi64_inline<typename BasicConfig<int64_t>::ISA_T>;
};

template <>
struct SubConfig<int32_t> : public BasicConfig<int32_t> {
  constexpr static auto naive_op = NaiveSub<int32_t>;
  constexpr static auto isa_op =
      sub_epi32_inline<typename BasicConfig<int32_t>::ISA_T>;
};

template <>
struct SubConfig<int64_t> : public BasicConfig<int64_t> {
  constexpr static auto naive_op = NaiveSub<int64_t>;
  constexpr static auto isa_op =
      sub_epi64_inline<typename BasicConfig<int64_t>::ISA_T>;
};

template <>
struct MulConfig<int32_t> : public BasicConfig<int32_t> {
  constexpr static auto naive_op = NaiveMul<int32_t>;
  constexpr static auto isa_op =
      mul_epi32_inline<typename BasicConfig<int32_t>::ISA_T>;
};

template <>
struct MulConfig<int64_t> : public BasicConfig<int64_t> {
  constexpr static auto naive_op = NaiveMul<int64_t>;
  constexpr static typename BasicConfig<int64_t>::ISA_T (*isa_op)(
      const typename BasicConfig<int64_t>::ISA_T,
      const typename BasicConfig<int64_t>::ISA_T) = nullptr;
};

template <>
struct MaxConfig<int32_t> : public BasicConfig<int32_t> {
  constexpr static auto naive_op = NaiveMax<int32_t>;
  constexpr static auto isa_op =
      max_epi32_inline<typename BasicConfig<int32_t>::ISA_T>;
};

template <>
struct MaxConfig<int64_t> : public BasicConfig<int64_t> {
  constexpr static auto naive_op = NaiveMax<int64_t>;
  constexpr static typename BasicConfig<int64_t>::ISA_T (*isa_op)(
      const typename BasicConfig<int64_t>::ISA_T,
      const typename BasicConfig<int64_t>::ISA_T) = nullptr;
};

template <>
struct MinConfig<int32_t> : public BasicConfig<int32_t> {
  constexpr static auto naive_op = NaiveMin<int32_t>;
  constexpr static auto isa_op =
      min_epi32_inline<typename BasicConfig<int32_t>::ISA_T>;
};

template <>
struct MinConfig<int64_t> : public BasicConfig<int64_t> {
  constexpr static auto naive_op = NaiveMin<int64_t>;
  constexpr static typename BasicConfig<int64_t>::ISA_T (*isa_op)(
      const typename BasicConfig<int64_t>::ISA_T,
      const typename BasicConfig<int64_t>::ISA_T) = nullptr;
};

// mod has no isa version and float version
template <>
struct ModConfig<int32_t> : public BasicConfig<int32_t> {
  constexpr static auto naive_op = NaiveMod<int32_t>;
  constexpr static typename BasicConfig<int32_t>::ISA_T (*isa_op)(
      const typename BasicConfig<int32_t>::ISA_T,
      const typename BasicConfig<int32_t>::ISA_T) = nullptr;
};

template <>
struct ModConfig<int64_t> : public BasicConfig<int64_t> {
  constexpr static auto naive_op = NaiveMod<int64_t>;
  constexpr static typename BasicConfig<int64_t>::ISA_T (*isa_op)(
      const typename BasicConfig<int64_t>::ISA_T,
      const typename BasicConfig<int64_t>::ISA_T) = nullptr;
};

// div except float has no isa version
template <>
struct DivConfig<int32_t> : public BasicConfig<int32_t> {
  constexpr static auto naive_op = NaiveDiv<int32_t>;
  constexpr static typename BasicConfig<int32_t>::ISA_T (*isa_op)(
      const typename BasicConfig<int32_t>::ISA_T,
      const typename BasicConfig<int32_t>::ISA_T) = nullptr;
};

template <>
struct DivConfig<int64_t> : public BasicConfig<int64_t> {
  constexpr static auto naive_op = NaiveDiv<int64_t>;
  constexpr static typename BasicConfig<int64_t>::ISA_T (*isa_op)(
      const typename BasicConfig<int64_t>::ISA_T,
      const typename BasicConfig<int64_t>::ISA_T) = nullptr;
};

// floordiv has no isa version
template <>
struct FloorDivConfig<int32_t> : public BasicConfig<int32_t> {
  constexpr static auto naive_op = NaiveFloorDiv<int32_t>;
  constexpr static typename BasicConfig<int32_t>::ISA_T (*isa_op)(
      const typename BasicConfig<int32_t>::ISA_T,
      const typename BasicConfig<int32_t>::ISA_T) = nullptr;
};

template <>
struct FloorDivConfig<int64_t> : public BasicConfig<int64_t> {
  constexpr static auto naive_op = NaiveFloorDiv<int64_t>;
  constexpr static typename BasicConfig<int64_t>::ISA_T (*isa_op)(
      const typename BasicConfig<int64_t>::ISA_T,
      const typename BasicConfig<int64_t>::ISA_T) = nullptr;
};

template <>
struct FloorDivConfig<float> : public BasicConfig<float> {
  constexpr static auto naive_op = NaiveFloorDiv<float>;
  constexpr static typename BasicConfig<float>::ISA_T (*isa_op)(
      const typename BasicConfig<float>::ISA_T,
      const typename BasicConfig<float>::ISA_T) = nullptr;
};

// pow has no isa version
template <>
struct PowConfig<int32_t> : public BasicConfig<int32_t> {
  constexpr static auto naive_op = NaivePow<int32_t>;
  constexpr static typename BasicConfig<int32_t>::ISA_T (*isa_op)(
      const typename BasicConfig<int32_t>::ISA_T,
      const typename BasicConfig<int32_t>::ISA_T) = nullptr;
};

template <>
struct PowConfig<int64_t> : public BasicConfig<int64_t> {
  constexpr static auto naive_op = NaivePow<int64_t>;
  constexpr static typename BasicConfig<int64_t>::ISA_T (*isa_op)(
      const typename BasicConfig<int64_t>::ISA_T,
      const typename BasicConfig<int64_t>::ISA_T) = nullptr;
};

template <>
struct PowConfig<float> : public BasicConfig<float> {
  constexpr static auto naive_op = NaivePow<float>;
  constexpr static typename BasicConfig<float>::ISA_T (*isa_op)(
      const typename BasicConfig<float>::ISA_T,
      const typename BasicConfig<float>::ISA_T) = nullptr;
};

// Active only support float version
template <class DataType>
struct ActiveConfig<ActiveType::NO_ACTIVE, DataType> {
  constexpr static DataType (*naive_active)(DataType) = nullptr;
  constexpr static typename BasicConfig<DataType>::ISA_T (*isa_active)(
      const typename BasicConfig<DataType>::ISA_T) = nullptr;
  constexpr static bool has_active = false;
};

#if defined(__AVX__)
namespace forward_avx = paddle::lite::x86::math::detail::forward::avx;

template <>
struct ActiveConfig<ActiveType::RELU, float> {
  constexpr static float (*naive_active)(float) = NaiveRelu<float>;
  constexpr static __m256 (*isa_active)(const __m256) = forward_avx::Relu;
  constexpr static bool has_active = true;
};

template <>
struct ActiveConfig<ActiveType::TANH, float> {
  constexpr static float (*naive_active)(float) = NaiveTanh<float>;
  constexpr static __m256 (*isa_active)(const __m256) = forward_avx::Tanh;
  constexpr static bool has_active = true;
};

template <>
struct ActiveConfig<ActiveType::SIGMOID, float> {
  constexpr static float (*naive_active)(float) = NaiveSigmoid<float>;
  constexpr static __m256 (*isa_active)(const __m256) = forward_avx::Sigmoid;
  constexpr static bool has_active = true;
};
#elif defined(__SSE4_2__)
__m128 _mm_relu_ps(const __m128& a);

template <>
struct ActiveConfig<ActiveType::RELU, float> {
  constexpr static float (*naive_active)(float) = NaiveRelu<float>;
  constexpr static __m128 (*isa_active)(const __m128) = _mm_relu_ps;
  constexpr static bool has_active = true;
};

// SSE has no tanh and sigmoid for now
template <>
struct ActiveConfig<ActiveType::TANH, float> {
  constexpr static float (*naive_active)(float) = NaiveTanh<float>;
  constexpr static __m128 (*isa_active)(const __m128) = nullptr;
  constexpr static bool has_active = true;
};

template <>
struct ActiveConfig<ActiveType::SIGMOID, float> {
  constexpr static float (*naive_active)(float) = NaiveSigmoid<float>;
  constexpr static __m128 (*isa_active)(const __m128) = nullptr;
  constexpr static bool has_active = true;
};
#endif

// fuse-activation doesn't support int32 and int64 type
template <>
struct ActiveConfig<ActiveType::RELU, int32_t> {
  constexpr static int32_t (*naive_active)(int32_t) = nullptr;
  constexpr static typename BasicConfig<int32_t>::ISA_T (*isa_active)(
      const typename BasicConfig<int32_t>::ISA_T) = nullptr;
  constexpr static bool has_active = false;
};

template <>
struct ActiveConfig<ActiveType::TANH, int32_t> {
  constexpr static int32_t (*naive_active)(int32_t) = nullptr;
  constexpr static typename BasicConfig<int32_t>::ISA_T (*isa_active)(
      const typename BasicConfig<int32_t>::ISA_T) = nullptr;
  constexpr static bool has_active = false;
};

template <>
struct ActiveConfig<ActiveType::SIGMOID, int32_t> {
  constexpr static int32_t (*naive_active)(int32_t) = nullptr;
  constexpr static typename BasicConfig<int32_t>::ISA_T (*isa_active)(
      const typename BasicConfig<int32_t>::ISA_T) = nullptr;
  constexpr static bool has_active = false;
};

template <>
struct ActiveConfig<ActiveType::RELU, int64_t> {
  constexpr static int64_t (*naive_active)(int64_t) = nullptr;
  constexpr static typename BasicConfig<int64_t>::ISA_T (*isa_active)(
      const typename BasicConfig<int64_t>::ISA_T) = nullptr;
  constexpr static bool has_active = false;
};

template <>
struct ActiveConfig<ActiveType::TANH, int64_t> {
  constexpr static int64_t (*naive_active)(int64_t) = nullptr;
  constexpr static typename BasicConfig<int64_t>::ISA_T (*isa_active)(
      const typename BasicConfig<int64_t>::ISA_T) = nullptr;
  constexpr static bool has_active = false;
};

template <>
struct ActiveConfig<ActiveType::SIGMOID, int64_t> {
  constexpr static int64_t (*naive_active)(int64_t) = nullptr;
  constexpr static typename BasicConfig<int64_t>::ISA_T (*isa_active)(
      const typename BasicConfig<int64_t>::ISA_T) = nullptr;
  constexpr static bool has_active = false;
};

// avoid compling error: xxx_address will never be null
static bool condition_one(void* isa_op, void* naive_op) {
  return ((isa_op != nullptr) && (naive_op != nullptr));
}

static bool condition_two(void* isa_op, void* naive_op) {
  return ((isa_op == nullptr) && (naive_op != nullptr));
}

static bool condition_three(void* isa_act) { return (isa_act != nullptr); }

// Fuse-Activation only supports relu, sigmoid and tanh for AVX instruction,
// relu for SSE instruction, the others run naive functions instead.
template <class Config, bool IS_X_SINGLE, bool IS_Y_SINGLE>
void do_isa_elementwise(const typename Config::T* dinx,
                        const typename Config::T* diny,
                        typename Config::T* dout,
                        int num) {
  static_assert((IS_X_SINGLE && IS_Y_SINGLE) != true,
                "X and Y could not be both single");
  using T = typename Config::T;
  using ISA_T = typename Config::ISA_T;
  using LD_T = typename Config::LD_T;
  constexpr auto isa_dup = Config::isa_dup;
  constexpr auto isa_ld = Config::isa_ld;
  constexpr auto isa_st = Config::isa_str;
  constexpr auto isa_op = Config::isa_op;
  constexpr auto naive_op = Config::naive_op;
  constexpr auto isa_act = Config::isa_active;
  constexpr auto naive_active = Config::naive_active;
  constexpr auto has_active = Config::has_active;
  constexpr int element_num = sizeof(ISA_T) / sizeof(T);
  int cnt = num / element_num;
  int remain = num % element_num;

  auto dinx_ptr = dinx;
  auto diny_ptr = diny;
  auto dout_ptr = dout;

  // avoid compiling error
  bool condition1 = condition_one(reinterpret_cast<void*>(isa_op),
                                  reinterpret_cast<void*>(naive_op));
  bool condition2 = condition_two(reinterpret_cast<void*>(isa_op),
                                  reinterpret_cast<void*>(naive_op));
  bool condition3 = condition_three(reinterpret_cast<void*>(isa_act));

  if (condition1) {
    ISA_T rbx, rby;
    if (IS_X_SINGLE) {
      rbx = isa_dup(*dinx);
    }
    if (IS_Y_SINGLE) {
      rby = isa_dup(*diny);
    }

    for (int i = 0; i < cnt; i++) {
      ISA_T dinx0, diny0, doutz0;
      if (!IS_X_SINGLE) {
        dinx0 = isa_ld(reinterpret_cast<const LD_T*>(dinx_ptr));
        dinx_ptr += element_num;
      }
      if (!IS_Y_SINGLE) {
        diny0 = isa_ld(reinterpret_cast<const LD_T*>(diny_ptr));
        diny_ptr += element_num;
      }
      if (IS_X_SINGLE && !IS_Y_SINGLE) {
        doutz0 = isa_op(rbx, diny0);
      } else if (!IS_X_SINGLE && IS_Y_SINGLE) {
        doutz0 = isa_op(dinx0, rby);
      } else if (!IS_X_SINGLE && !IS_Y_SINGLE) {
        doutz0 = isa_op(dinx0, diny0);
      }

      if (has_active && condition3) {
        doutz0 = isa_act(doutz0);
      } else if (has_active) {
        T* tmp_data = reinterpret_cast<T*>(&doutz0);
        for (int ii = 0; ii < element_num; ii++) {
          tmp_data[ii] = naive_active(tmp_data[ii]);
        }
      }
      isa_st(reinterpret_cast<LD_T*>(dout_ptr), doutz0);
      dout_ptr += element_num;
    }
    if (remain > 0) {
      for (int p = 0; p < remain; p++) {
        auto tmp = naive_op(*dinx_ptr, *diny_ptr);
        if (has_active) {
          tmp = naive_active(tmp);
        }
        *dout_ptr = tmp;
        dout_ptr++;
        if (!IS_X_SINGLE) {
          dinx_ptr++;
        }
        if (!IS_Y_SINGLE) {
          diny_ptr++;
        }
      }
    }
  } else if (condition2) {
    for (int p = 0; p < num; p++) {
      auto tmp = naive_op(*dinx_ptr, *diny_ptr);
      if (has_active) {
        tmp = naive_active(tmp);
      }
      *dout_ptr = tmp;
      dout_ptr++;
      if (!IS_X_SINGLE) {
        dinx_ptr++;
      }
      if (!IS_Y_SINGLE) {
        diny_ptr++;
      }
    }
  } else {
    LOG(FATAL) << "do_isa_elementwise has no op function to call.";
  }
}

template <class Config>
void elementwise_one_to_range(const typename Config::T* dinx,
                              const typename Config::T* diny,
                              typename Config::T* dout,
                              int num) {
  do_isa_elementwise<Config, true, false>(dinx, diny, dout, num);
}

template <class Config>
void elementwise_range_to_one(const typename Config::T* dinx,
                              const typename Config::T* diny,
                              typename Config::T* dout,
                              int num) {
  do_isa_elementwise<Config, false, true>(dinx, diny, dout, num);
}

template <class Config>
void elementwise_range_to_range(const typename Config::T* dinx,
                                const typename Config::T* diny,
                                typename Config::T* dout,
                                int num) {
  do_isa_elementwise<Config, false, false>(dinx, diny, dout, num);
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
