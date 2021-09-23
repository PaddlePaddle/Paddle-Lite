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

#include "lite/backends/x86/math/power.h"
#include <immintrin.h>
#include <cmath>
#include "lite/backends/x86/math/avx/avx_mathfuns.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <>
void power<float>(const float* din,
                  float* dout,
                  const int num,
                  float scale_,
                  float shift_,
                  float factor_) {
  int cnt = num >> 4;
  int remain = num % 16;
  bool _do_power = true;
  bool _do_scale = true;
  bool _do_shift = true;
  int rem_cnt = remain >> 2;
  int rem_rem = remain & 3;
  if (fabsf(factor_ - 1.f) < 1e-6f) {
    _do_power = false;
  }
  if (fabsf(scale_ - 1.f) < 1e-6f) {
    _do_scale = false;
  }
  if (fabsf(shift_ - 0.f) < 1e-6f) {
    _do_shift = false;
  }
#ifdef __AVX__
  __m256 vscale_256 = _mm256_set1_ps(scale_);
  __m256 vshift_256 = _mm256_set1_ps(shift_);
  __m256 vfactor_256 = _mm256_set1_ps(factor_);
#endif
  __m128 vscale = _mm_set1_ps(scale_);
  __m128 vshift = _mm_set1_ps(shift_);
  float* ptr_out = dout;
  const float* ptr_in = din;
  if (_do_power) {
    for (int i = 0; i < cnt; i++) {
#ifdef __AVX__
      __m256 vin0 = _mm256_loadu_ps(ptr_in);
      __m256 vin1 = _mm256_loadu_ps(ptr_in + 8);
      ptr_in += 16;
      __m256 vsum0 = _mm256_mul_ps(vin0, vscale_256);
      __m256 vsum1 = _mm256_mul_ps(vin1, vscale_256);
      __m256 vres0 = _mm256_add_ps(vsum0, vshift_256);
      __m256 vres1 = _mm256_add_ps(vsum1, vshift_256);
      vres0 = pow256_ps(vres0, vfactor_256);
      vres1 = pow256_ps(vres1, vfactor_256);
      _mm256_storeu_ps(ptr_out, vres0);
      _mm256_storeu_ps(ptr_out + 8, vres1);
#else
      __m128 vin0 = _mm_loadu_ps(ptr_in);
      __m128 vin1 = _mm_loadu_ps(ptr_in + 4);
      __m128 vin2 = _mm_loadu_ps(ptr_in + 8);
      __m128 vin3 = _mm_loadu_ps(ptr_in + 12);
      __m128 vsum0 = _mm_mul_ps(vin0, vscale);
      __m128 vsum1 = _mm_mul_ps(vin1, vscale);
      __m128 vsum2 = _mm_mul_ps(vin2, vscale);
      __m128 vsum3 = _mm_mul_ps(vin3, vscale);
      __m128 vres0 = _mm_add_ps(vsum0, vshift);
      __m128 vres1 = _mm_add_ps(vsum1, vshift);
      __m128 vres2 = _mm_add_ps(vsum2, vshift);
      __m128 vres3 = _mm_add_ps(vsum3, vshift);

      ptr_in += 16;
      for (int j = 0; j < 4; j++) {
        ptr_out[j] = std::pow((reinterpret_cast<float*>(&vres0))[j], factor_);
        ptr_out[j + 4] =
            std::pow((reinterpret_cast<float*>(&vres1))[j], factor_);
        ptr_out[j + 8] =
            std::pow((reinterpret_cast<float*>(&vres2))[j], factor_);
        ptr_out[j + 12] =
            std::pow((reinterpret_cast<float*>(&vres3))[j], factor_);
      }
#endif
      ptr_out += 16;
    }
    for (int i = 0; i < rem_cnt; i++) {
      __m128 vin0 = _mm_loadu_ps(ptr_in);
      ptr_in += 4;
      __m128 vsum0 = _mm_mul_ps(vin0, vscale);
      __m128 vres0 = _mm_add_ps(vsum0, vshift);
      for (int j = 0; j < 4; j++) {
        ptr_out[j] = std::pow((reinterpret_cast<float*>(&vres0))[j], factor_);
      }
      ptr_out += 4;
    }
    for (int i = 0; i < rem_rem; i++) {
      ptr_out[0] = std::pow((ptr_in[0] * scale_ + shift_), factor_);
      ptr_in++;
      ptr_out++;
    }
  } else {
    for (int i = 0; i < cnt; i++) {
#ifdef __AVX__
      __m256 vin0 = _mm256_loadu_ps(ptr_in);
      __m256 vin1 = _mm256_loadu_ps(ptr_in + 8);
      ptr_in += 16;
      __m256 vsum0 = _mm256_mul_ps(vin0, vscale_256);
      __m256 vsum1 = _mm256_mul_ps(vin1, vscale_256);
      __m256 vres0 = _mm256_add_ps(vsum0, vshift_256);
      __m256 vres1 = _mm256_add_ps(vsum1, vshift_256);
      _mm256_storeu_ps(ptr_out, vres0);
      _mm256_storeu_ps(ptr_out + 8, vres1);
      ptr_out += 16;
#else
      __m128 vin0 = _mm_loadu_ps(ptr_in);
      __m128 vin1 = _mm_loadu_ps(ptr_in + 4);
      __m128 vin2 = _mm_loadu_ps(ptr_in + 8);
      __m128 vin3 = _mm_loadu_ps(ptr_in + 12);
      __m128 vsum0 = _mm_mul_ps(vin0, vscale);
      __m128 vsum1 = _mm_mul_ps(vin1, vscale);
      __m128 vsum2 = _mm_mul_ps(vin2, vscale);
      __m128 vsum3 = _mm_mul_ps(vin3, vscale);
      __m128 vres0 = _mm_add_ps(vsum0, vshift);
      __m128 vres1 = _mm_add_ps(vsum1, vshift);
      __m128 vres2 = _mm_add_ps(vsum2, vshift);
      __m128 vres3 = _mm_add_ps(vsum3, vshift);

      ptr_in += 16;
      _mm_storeu_ps(ptr_out, vres0);
      _mm_storeu_ps(ptr_out + 4, vres1);
      _mm_storeu_ps(ptr_out + 8, vres2);
      _mm_storeu_ps(ptr_out + 12, vres3);
      ptr_out += 16;
#endif
    }
    for (int i = 0; i < rem_cnt; i++) {
      __m128 vin0 = _mm_loadu_ps(ptr_in);
      ptr_in += 4;
      __m128 vsum0 = _mm_mul_ps(vin0, vscale);
      __m128 vres0 = _mm_add_ps(vsum0, vshift);

      _mm_storeu_ps(ptr_out, vres0);
      ptr_out += 4;
    }
    for (int i = 0; i < rem_rem; i++) {
      ptr_out[0] = ptr_in[0] * scale_ + shift_;
      ptr_in++;
      ptr_out++;
    }
  }
}

} /* namespace math */
} /* namespace x86 */
} /* namespace lite */
} /* namespace paddle */
