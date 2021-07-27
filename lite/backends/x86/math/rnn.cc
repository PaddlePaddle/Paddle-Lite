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

#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE4_2__)
#include <xmmintrin.h>
#endif

#include "lite/backends/x86/math/detail/activation_functions.h"
#include "lite/backends/x86/math/rnn.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

using namespace paddle::lite::x86::math::detail::forward;

void vector_dot(
    float* out, const float* in, const float* v1, int size, const float* v2) {
#if defined(__AVX__)
  __m256 vec_in, vec_v1, vec_v2;
#endif
#if defined(__SSE4_2__)
  __m128 vec_in_128, vec_v1_128, vec_v2_128;
#endif

  int i = 0;
  if (nullptr == v2) {
    i = 0;

// in_out * v1
#if defined(__AVX__)
    for (; i + 7 < size; i += 8) {
      vec_in = _mm256_loadu_ps(in + i);
      vec_v1 = _mm256_loadu_ps(v1 + i);
      _mm256_storeu_ps(out + i, _mm256_mul_ps(vec_in, vec_v1));
    }
    _mm256_zeroupper();
#endif
#if defined(__SSE4_2__)
    for (; i + 3 < size; i += 4) {
      vec_in_128 = _mm_loadu_ps(in + i);
      vec_v1_128 = _mm_loadu_ps(v1 + i);
      _mm_storeu_ps(out + i, _mm_mul_ps(vec_in_128, vec_v1_128));
    }
#endif
    for (; i < size; i++) {
      out[i] = in[i] * v1[i];
    }
  } else {
    i = 0;

// in_out + v1 * v2
#if defined(__AVX__) && defined(__FMA__)
    for (; i + 7 < size; i += 8) {
      vec_in = _mm256_loadu_ps(in + i);
      vec_v1 = _mm256_loadu_ps(v1 + i);
      vec_v2 = _mm256_loadu_ps(v2 + i);
      _mm256_storeu_ps(out + i, _mm256_fmadd_ps(vec_v2, vec_v1, vec_in));
    }
    for (; i + 3 < size; i += 4) {
      vec_in_128 = _mm_loadu_ps(in + i);
      vec_v1_128 = _mm_loadu_ps(v1 + i);
      vec_v2_128 = _mm_loadu_ps(v2 + i);
      _mm_storeu_ps(out + i, _mm_fmadd_ps(vec_v2_128, vec_v1_128, vec_in_128));
    }
#endif
    for (; i < size; i++) {
      out[i] = in[i] + v1[i] * v2[i];
    }
  }
}

template <>
void act_relu<float>(const float* din, float* dout, int size, int threads) {
  int i = 0;

#ifdef __AVX__
  for (; i + 7 < size; i += 8) {
    __m256 a = _mm256_loadu_ps(din + i);
    _mm256_storeu_ps(dout + i, avx::Relu(a));
  }
#endif
  for (; i < size; i++) {
    dout[i] = Relu<float>(din[i]);
  }
}

template <>
void act_sigmoid<float>(const float* din, float* dout, int size, int threads) {
  int i = 0;

#ifdef __AVX__
  for (; i + 7 < size; i += 8) {
    __m256 a = _mm256_loadu_ps(din + i);
    _mm256_storeu_ps(dout + i, avx::Sigmoid(a));
  }
#endif
  for (; i < size; i++) {
    dout[i] = Sigmoid<float>(din[i]);
  }
}

template <>
void act_tanh<float>(const float* din, float* dout, int size, int threads) {
  int i = 0;

#ifdef __AVX__
  for (; i + 7 < size; i += 8) {
    __m256 a = _mm256_loadu_ps(din + i);
    _mm256_storeu_ps(dout + i, avx::Tanh(a));
  }
#endif
  for (; i < size; i++) {
    dout[i] = Tanh<float>(din[i]);
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle