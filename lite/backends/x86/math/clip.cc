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

#include "lite/backends/x86/math/clip.h"
#include <immintrin.h>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {
template <>
void clip<float>(
    const float* din, float* dout, const int num, float max_, float min_) {
  int cnt = num >> 4;
  int remain = num % 16;
  int rem_cnt = remain >> 2;
  int rem_rem = remain & 3;
  float* ptr_out = dout;
  const float* ptr_in = din;
#ifdef __AVX__
  __m256 max_256 = _mm256_set1_ps(max_);
  __m256 min_256 = _mm256_set1_ps(min_);
#endif
  __m128 vmax = _mm_set1_ps(max_);
  __m128 vmin = _mm_set1_ps(min_);
  for (int i = 0; i < cnt; i++) {
#ifdef __AVX__
    __m256 vin0 = _mm256_loadu_ps(ptr_in);
    __m256 vin1 = _mm256_loadu_ps(ptr_in + 8);
    vin0 = _mm256_min_ps(_mm256_max_ps(vin0, min_256), max_256);
    vin1 = _mm256_min_ps(_mm256_max_ps(vin1, min_256), max_256);
    _mm256_storeu_ps(ptr_out, vin0);
    _mm256_storeu_ps(ptr_out + 8, vin1);
#else
    __m128 vin0 = _mm_loadu_ps(ptr_in);
    __m128 vin1 = _mm_loadu_ps(ptr_in + 4);
    __m128 vin2 = _mm_loadu_ps(ptr_in + 8);
    __m128 vin3 = _mm_loadu_ps(ptr_in + 12);

    vin0 = _mm_min_ps(_mm_max_ps(vin0, vmin), vmax);
    vin1 = _mm_min_ps(_mm_max_ps(vin1, vmin), vmax);
    vin2 = _mm_min_ps(_mm_max_ps(vin2, vmin), vmax);
    vin3 = _mm_min_ps(_mm_max_ps(vin3, vmin), vmax);

    _mm_storeu_ps(ptr_out, vin0);
    _mm_storeu_ps(ptr_out + 4, vin1);
    _mm_storeu_ps(ptr_out + 8, vin2);
    _mm_storeu_ps(ptr_out + 12, vin3);
#endif
    ptr_in += 16;
    ptr_out += 16;
  }
  for (int i = 0; i < rem_cnt; i++) {
    __m128 vin0 = _mm_loadu_ps(ptr_in);
    vin0 = _mm_min_ps(_mm_max_ps(vin0, vmin), vmax);
    _mm_storeu_ps(ptr_out, vin0);
    ptr_in += 4;
    ptr_out += 4;
  }
  for (int i = 0; i < rem_rem; i++) {
    float tmp = ptr_in[0] > min_ ? ptr_in[0] : min_;
    ptr_out[0] = tmp < max_ ? tmp : max_;
    ptr_in++;
    ptr_out++;
  }
}

} /* namespace math */
} /* namespace x86 */
} /* namespace lite */
} /* namespace paddle */
