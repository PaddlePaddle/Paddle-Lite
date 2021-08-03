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

#include "lite/backends/x86/math/activation.h"

#ifdef __AVX__
#include <immintrin.h>
#include "lite/backends/x86/math/avx_mathfuns.h"
#else
#include <emmintrin.h>
#include <smmintrin.h>
#endif

#include <cmath>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <>
void mish(const float* din, float* dout, int size, float threshold) {
#ifdef __AVX__
  int cnt = size >> 3;
  int remain = size & 7;
#else
  int cnt = size >> 2;
  int remain = size & 3;
#endif

#ifdef __AVX__
  __m256 vthreshold = _mm256_set1_ps(threshold);
  __m256 vone = _mm256_set1_ps(1.f);
  __m256 vtwo = _mm256_set1_ps(2.f);
  __m256 minus_vthreshold = _mm256_set1_ps(-threshold);
  for (int i = 0; i < cnt; i++) {
    __m256 vx0 = _mm256_loadu_ps(din);

    __m256 gt_0 = _mm256_cmp_ps(vx0, vthreshold, _CMP_GT_OS);
    __m256 lt_0 = _mm256_cmp_ps(vx0, minus_vthreshold, _CMP_LT_OS);

    __m256 vleftx0 = exp256_ps(vx0);

    __m256 vmiddle_temp = _mm256_add_ps(vleftx0, vone);  // ln(1+e^x)
    __m256 vmiddlex0 = log256_ps(vmiddle_temp);

    __m256 sp0 = _mm256_blendv_ps(vmiddlex0, vx0, gt_0);
    sp0 = _mm256_blendv_ps(sp0, vleftx0, lt_0);

    __m256 exp_sp0 = exp256_ps(_mm256_mul_ps(sp0, vtwo));

    __m256 exp_sum0 = _mm256_add_ps(exp_sp0, vone);
    __m256 exp_diff0 = _mm256_sub_ps(exp_sp0, vone);
    __m256 tanh = _mm256_div_ps(exp_diff0, exp_sum0);
    __m256 res0 = _mm256_mul_ps(vx0, tanh);

    _mm256_storeu_ps(dout, res0);
    dout += 8;
    din += 8;
  }

#else

  __m128 vthreshold = _mm_set1_ps(threshold);
  __m128 vone = _mm_set1_ps(1.f);
  __m128 minus_vthreshold = _mm_set1_ps(-threshold);
  for (int i = 0; i < cnt; i++) {
    __m128 vx0 = _mm_loadu_ps(din);

    __m128 gt_0 = _mm_cmpgt_ps(vx0, vthreshold);
    __m128 lt_0 = _mm_cmplt_ps(vx0, minus_vthreshold);

    __m128 data0 = _mm_min_ps(vx0, _mm_set1_ps(70.00008f));
    data0 = _mm_max_ps(data0, _mm_set1_ps(-70.00008f));

    __m128 vleftx0;
    vleftx0[0] = std::exp(data0[0]);
    vleftx0[1] = std::exp(data0[1]);
    vleftx0[2] = std::exp(data0[2]);
    vleftx0[3] = std::exp(data0[3]);

    __m128 vmiddlex0;
    vmiddlex0[0] = std::log1p(vleftx0[0]);
    vmiddlex0[1] = std::log1p(vleftx0[1]);
    vmiddlex0[2] = std::log1p(vleftx0[2]);
    vmiddlex0[3] = std::log1p(vleftx0[3]);

    __m128 sp0 = _mm_blendv_ps(vmiddlex0, vx0, gt_0);
    sp0 = _mm_blendv_ps(sp0, vleftx0, lt_0);

    sp0 = _mm_min_ps(sp0, _mm_set1_ps(70.00008f));
    sp0 = _mm_max_ps(sp0, _mm_set1_ps(-70.00008f));

    __m128 exp_sp0;
    exp_sp0[0] = std::exp(2 * sp0[0]);
    exp_sp0[1] = std::exp(2 * sp0[1]);
    exp_sp0[2] = std::exp(2 * sp0[2]);
    exp_sp0[3] = std::exp(2 * sp0[3]);

    __m128 exp_sum0 = _mm_add_ps(exp_sp0, vone);
    __m128 exp_diff0 = _mm_sub_ps(exp_sp0, vone);
    __m128 tanh = _mm_div_ps(exp_diff0, exp_sum0);
    __m128 res0 = _mm_mul_ps(vx0, tanh);

    _mm_storeu_ps(dout, res0);
    dout += 4;
    din += 4;
  }

#endif

  for (int i = 0; i < remain; i++) {
    float x = din[i];
    float sp = 0.0f;
    if (threshold > 0 && x > threshold)
      sp = x;
    else if (threshold > 0 && x < -threshold)
      sp = std::exp(x);
    else
      sp = std::log1p(std::exp(x));
    dout[i] = x * std::tanh(sp);
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
