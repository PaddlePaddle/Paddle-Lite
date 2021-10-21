//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include "lite/backends/x86/cpu_info.h"
#ifdef __AVX__
namespace paddle {
namespace lite {
namespace x86 {
namespace math {
/* __m128 is ugly to write */
typedef __m256 v8sf;   // vector of 8 float (avx)
typedef __m256i v8si;  // vector of 8 int   (avx)
typedef __m128i v4si;  // vector of 8 int   (avx)
v8sf log256_ps(v8sf x);
v8sf exp256_ps(v8sf x);
v8sf pow256_ps(v8sf x, v8sf y);
v8sf sin256_ps(v8sf x);
v8sf cos256_ps(v8sf x);
void sincos256_ps(v8sf x, v8sf *s, v8sf *c);

// FMA support
#ifndef __AVX2__
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))

#define _mm256_permutevar8x32_ps(a, b)                                       \
  _mm256_setr_ps(                                                            \
      *(reinterpret_cast<float *>(&a)) + *(reinterpret_cast<int *>(&b)),     \
      *(reinterpret_cast<float *>(&a)) + *(reinterpret_cast<int *>(&b) + 1), \
      *(reinterpret_cast<float *>(&a)) + *(reinterpret_cast<int *>(&b) + 2), \
      *(reinterpret_cast<float *>(&a)) + *(reinterpret_cast<int *>(&b) + 3), \
      *(reinterpret_cast<float *>(&a)) + *(reinterpret_cast<int *>(&b) + 4), \
      *(reinterpret_cast<float *>(&a)) + *(reinterpret_cast<int *>(&b) + 5), \
      *(reinterpret_cast<float *>(&a)) + *(reinterpret_cast<int *>(&b) + 6), \
      *(reinterpret_cast<float *>(&a)) + *(reinterpret_cast<int *>(&b) + 7))
#endif
#ifndef __AVX512__
#define _mm_loadu_epi8(ptr) \
  _mm_loadu_si128(reinterpret_cast<__m128i const *>(ptr))
#endif

#ifndef __AVX512__
#define _mm256_loadu_epi8(ptr) \
  _mm256_loadu_si256(reinterpret_cast<__m256i const *>(ptr))
#define _mm_loadu_epi8(ptr) \
  _mm_loadu_si128(reinterpret_cast<__m128i const *>(ptr))
#define _mm256_storeu_epi8(ptr) \
  _mm256_storeu_si256(reinterpret_cast<__m256i const *>(ptr))
#define _mm_storeu_epi8(ptr) \
  _mm_storeu_si128(reinterpret_cast<__m128i const *>(ptr))
#endif

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
#endif
