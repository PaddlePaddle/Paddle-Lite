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

#include "lite/backends/x86/math/calib.h"
#include <string.h>
#include <vector>
#include "lite/backends/x86/math/avx/avx_mathfuns.h"
#include "lite/backends/x86/math/saturate.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {
void fp32_to_int8(const float* din,
                  int8_t* dout,
                  const float* scale,
                  int axis_size,
                  int64_t outer_size,
                  int64_t inner_size) {
#ifdef __AVX__
  int cnt = inner_size >> 5;
  int remain = inner_size & 31;
#else
  int cnt = inner_size >> 4;
  int remain = inner_size & 15;
#endif
  int rem_cnt = remain >> 3;
  int rem_rem = remain & 7;
  int64_t loop_size = outer_size * axis_size;
#pragma omp parallel for
  for (int j = 0; j < loop_size; ++j) {
    float inv_scale = 1.f / scale[j % axis_size];
    __m128 vzero = _mm_set1_ps(-127.f);
    __m128 vscale = _mm_set1_ps(inv_scale);
    __m256 vzero_l = _mm256_set1_ps(-127.f);
    __m256 vscale_l = _mm256_set1_ps(inv_scale);
    const float* din_c = din + j * inner_size;
    int8_t* dout_c = dout + j * inner_size;
#ifdef __AVX__
    for (int i = 0; i < cnt; i++) {
      __m256 vin0 = _mm256_loadu_ps(din_c);
      __m256 vin1 = _mm256_loadu_ps(din_c + 8);
      __m256 vin2 = _mm256_loadu_ps(din_c + 16);
      __m256 vin3 = _mm256_loadu_ps(din_c + 24);
      __m256 vout0 = _mm256_mul_ps(vin0, vscale_l);
      __m256 vout1 = _mm256_mul_ps(vin1, vscale_l);
      __m256 vout2 = _mm256_mul_ps(vin2, vscale_l);
      __m256 vout3 = _mm256_mul_ps(vin3, vscale_l);
      vin0 = _mm256_blendv_ps(
          vzero_l, vout0, _mm256_cmp_ps(vout0, vzero_l, _CMP_GT_OS));
      vin1 = _mm256_blendv_ps(
          vzero_l, vout1, _mm256_cmp_ps(vout1, vzero_l, _CMP_GT_OS));
      vin2 = _mm256_blendv_ps(
          vzero_l, vout2, _mm256_cmp_ps(vout2, vzero_l, _CMP_GT_OS));
      vin3 = _mm256_blendv_ps(
          vzero_l, vout3, _mm256_cmp_ps(vout3, vzero_l, _CMP_GT_OS));
      // fp32->int32
      __m256i vres0 = _mm256_cvtps_epi32(vin0);
      __m256i vres1 = _mm256_cvtps_epi32(vin1);
      __m256i vres2 = _mm256_cvtps_epi32(vin2);
      __m256i vres3 = _mm256_cvtps_epi32(vin3);
      __m256i vres0_16 = _mm256_packs_epi32(vres0, vres0);
      __m256i vres1_16 = _mm256_packs_epi32(vres1, vres1);
      __m256i vres2_16 = _mm256_packs_epi32(vres2, vres2);
      __m256i vres3_16 = _mm256_packs_epi32(vres3, vres3);
      __m256i vres0_8 = _mm256_packs_epi16(vres0_16, vres0_16);
      __m256i vres1_8 = _mm256_packs_epi16(vres1_16, vres1_16);
      __m256i vres2_8 = _mm256_packs_epi16(vres2_16, vres2_16);
      __m256i vres3_8 = _mm256_packs_epi16(vres3_16, vres3_16);
      *(reinterpret_cast<int*>(dout_c)) = (reinterpret_cast<int*>(&vres0_8))[0];
      *(reinterpret_cast<int*>(dout_c + 4)) =
          (reinterpret_cast<int*>(&vres0_8))[4];
      *(reinterpret_cast<int*>(dout_c + 8)) =
          (reinterpret_cast<int*>(&vres1_8))[0];
      *(reinterpret_cast<int*>(dout_c + 12)) =
          (reinterpret_cast<int*>(&vres1_8))[4];
      *(reinterpret_cast<int*>(dout_c + 16)) =
          (reinterpret_cast<int*>(&vres2_8))[0];
      *(reinterpret_cast<int*>(dout_c + 20)) =
          (reinterpret_cast<int*>(&vres2_8))[4];
      *(reinterpret_cast<int*>(dout_c + 24)) =
          (reinterpret_cast<int*>(&vres3_8))[0];
      *(reinterpret_cast<int*>(dout_c + 28)) =
          (reinterpret_cast<int*>(&vres3_8))[4];
      din_c += 32;
      dout_c += 32;
    }
#else
    for (int i = 0; i < cnt; i++) {
      __m128 vin0 = _mm_loadu_ps(din_c);
      __m128 vin1 = _mm_loadu_ps(din_c + 4);
      __m128 vin2 = _mm_loadu_ps(din_c + 8);
      __m128 vin3 = _mm_loadu_ps(din_c + 12);
      __m128 vout0 = _mm_mul_ps(vin0, vscale);
      __m128 vout1 = _mm_mul_ps(vin1, vscale);
      __m128 vout2 = _mm_mul_ps(vin2, vscale);
      __m128 vout3 = _mm_mul_ps(vin3, vscale);
      vin0 = _mm_blendv_ps(vzero, vout0, _mm_cmp_ps(vout0, vzero, _CMP_GT_OS));
      vin1 = _mm_blendv_ps(vzero, vout1, _mm_cmp_ps(vout1, vzero, _CMP_GT_OS));
      vin2 = _mm_blendv_ps(vzero, vout2, _mm_cmp_ps(vout2, vzero, _CMP_GT_OS));
      vin3 = _mm_blendv_ps(vzero, vout3, _mm_cmp_ps(vout3, vzero, _CMP_GT_OS));
      // fp32->int32
      __m128i vres0 = _mm_cvtps_epi32(vin0);
      __m128i vres1 = _mm_cvtps_epi32(vin1);
      __m128i vres2 = _mm_cvtps_epi32(vin2);
      __m128i vres3 = _mm_cvtps_epi32(vin3);
      __m128i vres0_16 = _mm_packs_epi32(vres0, vres0);
      __m128i vres1_16 = _mm_packs_epi32(vres1, vres1);
      __m128i vres2_16 = _mm_packs_epi32(vres2, vres2);
      __m128i vres3_16 = _mm_packs_epi32(vres3, vres3);
      __m128i vres0_8 = _mm_packs_epi16(vres0_16, vres0_16);
      __m128i vres1_8 = _mm_packs_epi16(vres1_16, vres1_16);
      __m128i vres2_8 = _mm_packs_epi16(vres2_16, vres2_16);
      __m128i vres3_8 = _mm_packs_epi16(vres3_16, vres3_16);
      *(reinterpret_cast<int*>(dout_c)) = _mm_extract_epi32(vres0_8, 0);
      *(reinterpret_cast<int*>(dout_c + 4)) = _mm_extract_epi32(vres1_8, 0);
      *(reinterpret_cast<int*>(dout_c + 8)) = _mm_extract_epi32(vres2_8, 0);
      *(reinterpret_cast<int*>(dout_c + 12)) = _mm_extract_epi32(vres3_8, 0);
      din_c += 16;
      dout_c += 16;
    }
#endif
    for (int i = 0; i < rem_cnt; i++) {
      __m128 vin0 = _mm_loadu_ps(din_c);
      __m128 vout0 = _mm_mul_ps(vin0, vscale);
      vin0 = _mm_blendv_ps(vzero, vout0, _mm_cmp_ps(vout0, vzero, _CMP_GT_OS));
      // fp32->int32
      __m128i vres0 = _mm_cvtps_epi32(vin0);
      __m128i vres0_16 = _mm_packs_epi32(vres0, vres0);
      __m128i vres0_8 = _mm_packs_epi16(vres0_16, vres0_16);
      *(reinterpret_cast<int*>(dout_c)) = _mm_extract_epi32(vres0_8, 0);
      din_c += 8;
      dout_c += 8;
    }
    for (int i = 0; i < rem_rem; ++i) {
      dout_c[i] = saturate_cast<int8_t>(roundf(inv_scale * din_c[i]));
      dout_c[i] = dout_c[i] < -127 ? -127 : dout_c[i];
    }
  }
}

void int8_to_fp32(const int8_t* in,
                  float* out,
                  const float* scale,
                  int axis_size,
                  int64_t outer_size,
                  int64_t inner_size) {
#ifdef __AVX__
  int cnt = inner_size >> 5;
  int remain = inner_size & 31;
#else
  int cnt = inner_size >> 4;
  int remain = inner_size & 15;
#endif
  int rem_cnt = remain >> 2;
  int rem_rem = remain & 3;
  int64_t loop_size = axis_size * outer_size;
#pragma omp parallel for
  for (int64_t n = 0; n < loop_size; ++n) {
    float in_scale = scale[n % axis_size];
    const int8_t* din_c = in + n * inner_size;
    float* dout_c = out + n * inner_size;
    __m128 vscale = _mm_set1_ps(in_scale);
    __m256 vscale_l = _mm256_set1_ps(in_scale);

#ifdef __AVX__
    for (int i = 0; i < cnt; i++) {
      __m128i vin0 = _mm_loadu_epi8(din_c);
      __m128i vin1 = _mm_loadu_epi8(din_c + 8);
      __m128i vin2 = _mm_loadu_epi8(din_c + 16);
      __m128i vin3 = _mm_loadu_epi8(din_c + 24);
      // 8bits x 16 -> 32bits x 8
      __m256i v00 = _mm256_cvtepi8_epi32(vin0);
      __m256i v01 = _mm256_cvtepi8_epi32(vin1);
      __m256i v02 = _mm256_cvtepi8_epi32(vin2);
      __m256i v03 = _mm256_cvtepi8_epi32(vin3);
      // int32 -> fp32
      __m256 vout0 = _mm256_mul_ps(_mm256_cvtepi32_ps(v00), vscale_l);
      __m256 vout1 = _mm256_mul_ps(_mm256_cvtepi32_ps(v01), vscale_l);
      __m256 vout2 = _mm256_mul_ps(_mm256_cvtepi32_ps(v02), vscale_l);
      __m256 vout3 = _mm256_mul_ps(_mm256_cvtepi32_ps(v03), vscale_l);
      _mm256_storeu_ps(dout_c, vout0);
      _mm256_storeu_ps(dout_c + 8, vout1);
      _mm256_storeu_ps(dout_c + 16, vout2);
      _mm256_storeu_ps(dout_c + 24, vout3);
      din_c += 32;
      dout_c += 32;
    }
#else
    for (int i = 0; i < cnt; i++) {
      __m128i vin0 = _mm_loadu_epi8(din_c);
      __m128i vin1 = _mm_loadu_epi8(din_c + 4);
      __m128i vin2 = _mm_loadu_epi8(din_c + 8);
      __m128i vin3 = _mm_loadu_epi8(din_c + 12);
      // 8bits x 16 -> 32bits x 4
      __m128i v00 = _mm_cvtepi8_epi32(vin0);
      __m128i v01 = _mm_cvtepi8_epi32(vin1);
      __m128i v02 = _mm_cvtepi8_epi32(vin2);
      __m128i v03 = _mm_cvtepi8_epi32(vin3);
      // int32 -> fp32
      __m128 vout0 = _mm_mul_ps(_mm_cvtepi32_ps(v00), vscale);
      __m128 vout1 = _mm_mul_ps(mm_cvtepi32_ps(v01), vscale);
      __m128 vout2 = _mm_mul_ps(mm_cvtepi32_ps(v02), vscale);
      __m128 vout3 = _mm_mul_ps(mm_cvtepi32_ps(v03), vscale);
      _mm_storeu_ps(dout_c, vout0);
      _mm_storeu_ps(dout_c + 4, vout1);
      _mm_storeu_ps(dout_c + 8, vout2);
      _mm_storeu_ps(dout_c + 12, vout3);
      din_c += 16;
      dout_c += 16;
    }
#endif
    for (int i = 0; i < rem_cnt; i++) {
      __m128i vin0 = _mm_loadu_epi8(din_c);
      // 8bits x 16 -> 32bits x 4
      __m128i v00 = _mm_cvtepi8_epi32(vin0);
      // int32 -> fp32
      __m128 vout0 = _mm_mul_ps(_mm_cvtepi32_ps(v00), vscale);
      _mm_storeu_ps(dout_c, vout0);

      din_c += 4;
      dout_c += 4;
    }
    for (int i = 0; i < rem_rem; ++i) {
      dout_c[i] = in_scale * din_c[i];
    }
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
