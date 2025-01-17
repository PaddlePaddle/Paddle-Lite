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

#include "lite/backends/loongarch/math/calib.h"
#include <string.h>
#include <vector>
#include "lite/backends/loongarch/xxl.h"
#include "lite/backends/loongarch/math/include/mathfuns.h"
#include "lite/backends/loongarch/math/saturate.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {
void fp32_to_int8(const float* din,
                  int8_t* dout,
                  const float* scale,
                  int axis_size,
                  int64_t outer_size,
                  int64_t inner_size) {
#ifdef __loongarch_asx
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
#ifdef __loongarch_asx
    __m256 vzero_l = lasx_set1_f32(-127.f);
    __m256 vscale_l = lasx_set1_f32(inv_scale);
#endif
    __m128 vzero = lsx_set1_f32(-127.f);
    __m128 vscale = lsx_set1_f32(inv_scale);
    const float* din_c = din + j * inner_size;
    int8_t* dout_c = dout + j * inner_size;
#ifdef __loongarch_asx
    for (int i = 0; i < cnt; i++) {
      __m256 vin0 = lasx_loadu_f32(din_c);
      __m256 vin1 = lasx_loadu_f32(din_c + 8);
      __m256 vin2 = lasx_loadu_f32(din_c + 16);
      __m256 vin3 = lasx_loadu_f32(din_c + 24);
      __m256 vout0 = lasx_mul_f32(vin0, vscale_l);
      __m256 vout1 = lasx_mul_f32(vin1, vscale_l);
      __m256 vout2 = lasx_mul_f32(vin2, vscale_l);
      __m256 vout3 = lasx_mul_f32(vin3, vscale_l);
      vin0 = lasx_blendv_f32(
          vzero_l, vout0, lasx_xvfcmp_slt_s(vzero_l, vout0));
      vin1 = lasx_blendv_f32(
          vzero_l, vout1, lasx_xvfcmp_slt_s(vzero_l, vout1));
      vin2 = lasx_blendv_f32(
          vzero_l, vout2, lasx_xvfcmp_slt_s(vzero_l, vout2));
      vin3 = lasx_blendv_f32(
          vzero_l, vout3, lasx_xvfcmp_slt_s(vzero_l, vout3));
      // fp32->int32
      __m256i vres0 = lasx_cvtf32_i32(vin0);
      __m256i vres1 = lasx_cvtf32_i32(vin1);
      __m256i vres2 = lasx_cvtf32_i32(vin2);
      __m256i vres3 = lasx_cvtf32_i32(vin3);
      __m256i vres0_16 = lasx_packs_i32(vres0, vres0);
      __m256i vres1_16 = lasx_packs_i32(vres1, vres1);
      __m256i vres2_16 = lasx_packs_i32(vres2, vres2);
      __m256i vres3_16 = lasx_packs_i32(vres3, vres3);
      __m256i vres0_8 = lasx_packs_i16(vres0_16, vres0_16);
      __m256i vres1_8 = lasx_packs_i16(vres1_16, vres1_16);
      __m256i vres2_8 = lasx_packs_i16(vres2_16, vres2_16);
      __m256i vres3_8 = lasx_packs_i16(vres3_16, vres3_16);
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
      __m128 vin0 = lsx_loadu_f32(din_c);
      __m128 vin1 = lsx_loadu_f32(din_c + 4);
      __m128 vin2 = lsx_loadu_f32(din_c + 8);
      __m128 vin3 = lsx_loadu_f32(din_c + 12);
      __m128 vout0 = lsx_mul_f32(vin0, vscale);
      __m128 vout1 = lsx_mul_f32(vin1, vscale);
      __m128 vout2 = lsx_mul_f32(vin2, vscale);
      __m128 vout3 = lsx_mul_f32(vin3, vscale);
      vin0 = lsx_blendv_f32(vzero, vout0, lsx_vfcmp_slt_s(vzero, vout0));
      vin1 = lsx_blendv_f32(vzero, vout1, lsx_vfcmp_slt_s(vzero, vout1));
      vin2 = lsx_blendv_f32(vzero, vout2, lsx_vfcmp_slt_s(vzero, vout2));
      vin3 = lsx_blendv_f32(vzero, vout3, lsx_vfcmp_slt_s(vzero, vout3));
      // fp32->int32
      __m128i vres0 = lsx_cvtf32_i32(vin0);
      __m128i vres1 = lsx_cvtf32_i32(vin1);
      __m128i vres2 = lsx_cvtf32_i32(vin2);
      __m128i vres3 = lsx_cvtf32_i32(vin3);
      __m128i vres0_16 = lsx_packs_i32(vres0, vres0);
      __m128i vres1_16 = lsx_packs_i32(vres1, vres1);
      __m128i vres2_16 = lsx_packs_i32(vres2, vres2);
      __m128i vres3_16 = lsx_packs_i32(vres3, vres3);
      __m128i vres0_8 = lsx_packs_i16(vres0_16, vres0_16);
      __m128i vres1_8 = lsx_packs_i16(vres1_16, vres1_16);
      __m128i vres2_8 = lsx_packs_i16(vres2_16, vres2_16);
      __m128i vres3_8 = lsx_packs_i16(vres3_16, vres3_16);
      *(reinterpret_cast<int*>(dout_c)) = lsx_extract_i32(vres0_8, 0);
      *(reinterpret_cast<int*>(dout_c + 4)) = lsx_extract_i32(vres1_8, 0);
      *(reinterpret_cast<int*>(dout_c + 8)) = lsx_extract_i32(vres2_8, 0);
      *(reinterpret_cast<int*>(dout_c + 12)) = lsx_extract_i32(vres3_8, 0);
      din_c += 16;
      dout_c += 16;
    }
#endif
    for (int i = 0; i < rem_cnt; i++) {
      __m128 vin0 = lsx_loadu_f32(din_c);
      __m128 vout0 = lsx_mul_f32(vin0, vscale);
      vin0 = lsx_blendv_f32(vzero, vout0, lsx_vfcmp_slt_s(vzero, vout0));
      // fp32->int32
      __m128i vres0 = lsx_cvtf32_i32(vin0);
      __m128i vres0_16 = lsx_packs_i32(vres0, vres0);
      __m128i vres0_8 = lsx_packs_i16(vres0_16, vres0_16);
      *(reinterpret_cast<int*>(dout_c)) = lsx_extract_i32(vres0_8, 0);
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
#ifdef __loongarch_asx
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
#ifdef __loongarch_asx
    __m256 vscale_l = lasx_set1_f32(in_scale);
#endif
    __m128 vscale = lsx_set1_f32(in_scale);

#ifdef __loongarch_asx
    for (int i = 0; i < cnt; i++) {
      __m128i vin0 = lsx_loadu_epi8(din_c);
      __m128i vin1 = lsx_loadu_epi8(din_c + 8);
      __m128i vin2 = lsx_loadu_epi8(din_c + 16);
      __m128i vin3 = lsx_loadu_epi8(din_c + 24);
      // 8bits x 16 -> 32bits x 8
      __m256i v00 = lasx_cvti8_i32(vin0);
      __m256i v01 = lasx_cvti8_i32(vin1);
      __m256i v02 = lasx_cvti8_i32(vin2);
      __m256i v03 = lasx_cvti8_i32(vin3);
      // int32 -> fp32
      __m256 vout0 = lasx_mul_f32(lasx_cvti32_f32(v00), vscale_l);
      __m256 vout1 = lasx_mul_f32(lasx_cvti32_f32(v01), vscale_l);
      __m256 vout2 = lasx_mul_f32(lasx_cvti32_f32(v02), vscale_l);
      __m256 vout3 = lasx_mul_f32(lasx_cvti32_f32(v03), vscale_l);
      lasx_storeu_f32(dout_c, vout0);
      lasx_storeu_f32(dout_c + 8, vout1);
      lasx_storeu_f32(dout_c + 16, vout2);
      lasx_storeu_f32(dout_c + 24, vout3);
      din_c += 32;
      dout_c += 32;
    }
#else
    for (int i = 0; i < cnt; i++) {
      __m128i vin0 = lsx_loadu_epi8(din_c);
      __m128i vin1 = lsx_loadu_epi8(din_c + 4);
      __m128i vin2 = lsx_loadu_epi8(din_c + 8);
      __m128i vin3 = lsx_loadu_epi8(din_c + 12);
      // 8bits x 16 -> 32bits x 4
      __m128i v00 = lsx_cvti8_i32(vin0);
      __m128i v01 = lsx_cvti8_i32(vin1);
      __m128i v02 = lsx_cvti8_i32(vin2);
      __m128i v03 = lsx_cvti8_i32(vin3);
      // int32 -> fp32
      __m128 vout0 = lsx_mul_f32(lsx_cvti32_f32(v00), vscale);
      __m128 vout1 = lsx_mul_f32(lsx_cvti32_f32(v01), vscale);
      __m128 vout2 = lsx_mul_f32(lsx_cvti32_f32(v02), vscale);
      __m128 vout3 = lsx_mul_f32(lsx_cvti32_f32(v03), vscale);
      lsx_storeu_f32(dout_c, vout0);
      lsx_storeu_f32(dout_c + 4, vout1);
      lsx_storeu_f32(dout_c + 8, vout2);
      lsx_storeu_f32(dout_c + 12, vout3);
      din_c += 16;
      dout_c += 16;
    }
#endif
    for (int i = 0; i < rem_cnt; i++) {
      __m128i vin0 = lsx_loadu_epi8(din_c);
      // 8bits x 16 -> 32bits x 4
      __m128i v00 = lsx_cvti8_i32(vin0);
      // int32 -> fp32
      __m128 vout0 = lsx_mul_f32(lsx_cvti32_f32(v00), vscale);
      lsx_storeu_f32(dout_c, vout0);

      din_c += 4;
      dout_c += 4;
    }
    for (int i = 0; i < rem_rem; ++i) {
      dout_c[i] = in_scale * din_c[i];
    }
  }
}

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
