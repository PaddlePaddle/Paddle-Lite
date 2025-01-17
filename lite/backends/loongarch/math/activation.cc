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

#include "lite/backends/loongarch/math/activation.h"

#include "lite/backends/loongarch/xxl.h"
#ifdef __loongarch_asx
#include "lite/backends/loongarch/math/include/mathfuns.h"
#endif

#include <algorithm>
#include <cmath>

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

template <>
void mish(const float* din, float* dout, int size, float threshold) {
#ifdef __loongarch_asx
  int cnt = size >> 3;
  int remain = size & 7;
#else
  int cnt = size >> 2;
  int remain = size & 3;
#endif

#ifdef __loongarch_asx
  __m256 vthreshold = lasx_set1_f32(threshold);
  __m256 vone = lasx_set1_f32(1.f);
  __m256 vtwo = lasx_set1_f32(2.f);
  __m256 minus_vthreshold = lasx_set1_f32(-threshold);
  for (int i = 0; i < cnt; i++) {
    __m256 vx0 = lasx_loadu_f32(din);

    __m256 gt_0 = lasx_xvfcmp_slt_s(vthreshold, vx0);
    __m256 lt_0 = lasx_xvfcmp_slt_s(vx0, minus_vthreshold);

    __m256 vleftx0 = exp256_ps(vx0);

    __m256 vmiddle_temp = lasx_add_f32(vleftx0, vone);  // ln(1+e^x)
    __m256 vmiddlex0 = log256_ps(vmiddle_temp);

    __m256 sp0 = lasx_blendv_f32(vmiddlex0, vx0, gt_0);
    sp0 = lasx_blendv_f32(sp0, vleftx0, lt_0);

    __m256 exp_sp0 = exp256_ps(lasx_mul_f32(sp0, vtwo));

    __m256 exp_sum0 = lasx_add_f32(exp_sp0, vone);
    __m256 exp_diff0 = lasx_sub_f32(exp_sp0, vone);
    __m256 tanh = lasx_div_f32(exp_diff0, exp_sum0);
    __m256 res0 = lasx_mul_f32(vx0, tanh);

    lasx_storeu_f32(dout, res0);
    dout += 8;
    din += 8;
  }

#else

  __m128 vthreshold = lsx_set1_f32(threshold);
  __m128 vone = lsx_set1_f32(1.f);
  __m128 minus_vthreshold = lsx_set1_f32(-threshold);
  for (int i = 0; i < cnt; i++) {
    __m128 vx0 = lsx_loadu_f32(din);

    __m128 gt_0 = lsx_cmpgt_f32(vx0, vthreshold);
    __m128 lt_0 = lsx_cmplt_f32(vx0, minus_vthreshold);

    __m128 data0 = lsx_min_f32(vx0, lsx_set1_f32(70.00008f));
    data0 = lsx_max_f32(data0, lsx_set1_f32(-70.00008f));

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

    __m128 sp0 = lsx_blendv_f32(vmiddlex0, vx0, gt_0);
    sp0 = lsx_blendv_f32(sp0, vleftx0, lt_0);

    sp0 = lsx_min_f32(sp0, lsx_set1_f32(70.00008f));
    sp0 = lsx_max_f32(sp0, lsx_set1_f32(-70.00008f));

    __m128 exp_sp0;
    exp_sp0[0] = std::exp(2 * sp0[0]);
    exp_sp0[1] = std::exp(2 * sp0[1]);
    exp_sp0[2] = std::exp(2 * sp0[2]);
    exp_sp0[3] = std::exp(2 * sp0[3]);

    __m128 exp_sum0 = lsx_add_f32(exp_sp0, vone);
    __m128 exp_diff0 = lsx_sub_f32(exp_sp0, vone);
    __m128 tanh = lsx_div_f32(exp_diff0, exp_sum0);
    __m128 res0 = lsx_mul_f32(vx0, tanh);

    lsx_storeu_f32(dout, res0);
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

template <>
void hard_swish(const float* din,
                float* dout,
                int size,
                float scale,
                float offset,
                float threshold) {
#ifdef __loongarch_asx
  int cnt = size >> 5;
  int remain = size & 31;
  __m256 vec_zero = lasx_set1_f32(0.f);
  __m256 vec_scale = lasx_set1_f32(1.0 / scale);
  __m256 vec_threshold = lasx_set1_f32(threshold);
  __m256 vec_offset = lasx_set1_f32(offset);
#else
  int cnt = size >> 4;
  int remain = size & 15;
#endif
  __m128 vec_zero_128 = lsx_set1_f32(0.f);
  __m128 vec_scale_128 = lsx_set1_f32(1.0 / scale);
  __m128 vec_threshold_128 = lsx_set1_f32(threshold);
  __m128 vec_offset_128 = lsx_set1_f32(offset);
  int cnt_4 = remain >> 2;
  int rem_4 = remain & 3;
  for (int i = 0; i < cnt; i++) {
#ifdef __loongarch_asx
    __m256 vin0 = lasx_loadu_f32(din);
    __m256 vin1 = lasx_loadu_f32(din + 8);
    __m256 vin2 = lasx_loadu_f32(din + 16);
    __m256 vin3 = lasx_loadu_f32(din + 24);
    __m256 vadd0 = lasx_add_f32(vin0, vec_offset);
    __m256 vadd1 = lasx_add_f32(vin1, vec_offset);
    __m256 vadd2 = lasx_add_f32(vin2, vec_offset);
    __m256 vadd3 = lasx_add_f32(vin3, vec_offset);
    __m256 vsum0 = lasx_mul_f32(vin0, vec_scale);
    __m256 vsum1 = lasx_mul_f32(vin1, vec_scale);
    __m256 vsum2 = lasx_mul_f32(vin2, vec_scale);
    __m256 vsum3 = lasx_mul_f32(vin3, vec_scale);
    __m256 vres0 = lasx_min_f32(lasx_max_f32(vadd0, vec_zero), vec_threshold);
    __m256 vres1 = lasx_min_f32(lasx_max_f32(vadd1, vec_zero), vec_threshold);
    __m256 vres2 = lasx_min_f32(lasx_max_f32(vadd2, vec_zero), vec_threshold);
    __m256 vres3 = lasx_min_f32(lasx_max_f32(vadd3, vec_zero), vec_threshold);
    lasx_storeu_f32(dout, lasx_mul_f32(vres0, vsum0));
    lasx_storeu_f32(dout + 8, lasx_mul_f32(vres1, vsum1));
    lasx_storeu_f32(dout + 16, lasx_mul_f32(vres2, vsum2));
    lasx_storeu_f32(dout + 24, lasx_mul_f32(vres3, vsum3));
    din += 32;
    dout += 32;
#else
    __m128 vin0 = lsx_loadu_f32(din);
    __m128 vin1 = lsx_loadu_f32(din + 4);
    __m128 vin2 = lsx_loadu_f32(din + 8);
    __m128 vin3 = lsx_loadu_f32(din + 12);
    __m128 vadd0 = lsx_add_f32(vin0, vec_offset_128);
    __m128 vadd1 = lsx_add_f32(vin1, vec_offset_128);
    __m128 vadd2 = lsx_add_f32(vin2, vec_offset_128);
    __m128 vadd3 = lsx_add_f32(vin3, vec_offset_128);
    __m128 vsum0 = lsx_mul_f32(vin0, vec_scale_128);
    __m128 vsum1 = lsx_mul_f32(vin1, vec_scale_128);
    __m128 vsum2 = lsx_mul_f32(vin2, vec_scale_128);
    __m128 vsum3 = lsx_mul_f32(vin3, vec_scale_128);
    __m128 vres0 =
        lsx_min_f32(lsx_max_f32(vadd0, vec_zero_128), vec_threshold_128);
    __m128 vres1 =
        lsx_min_f32(lsx_max_f32(vadd1, vec_zero_128), vec_threshold_128);
    __m128 vres2 =
        lsx_min_f32(lsx_max_f32(vadd2, vec_zero_128), vec_threshold_128);
    __m128 vres3 =
        lsx_min_f32(lsx_max_f32(vadd3, vec_zero_128), vec_threshold_128);
    lsx_storeu_f32(dout, lsx_mul_f32(vres0, vsum0));
    lsx_storeu_f32(dout + 4, lsx_mul_f32(vres1, vsum1));
    lsx_storeu_f32(dout + 8, lsx_mul_f32(vres2, vsum2));
    lsx_storeu_f32(dout + 12, lsx_mul_f32(vres3, vsum3));
    din += 16;
    dout += 16;
#endif
  }
  for (int i = 0; i < cnt_4; i++) {
    __m128 vin0 = lsx_loadu_f32(din);
    __m128 vadd0 = lsx_add_f32(vin0, vec_offset_128);
    __m128 vsum0 = lsx_mul_f32(vin0, vec_scale_128);
    __m128 vres0 =
        lsx_min_f32(lsx_max_f32(vadd0, vec_zero_128), vec_threshold_128);
    lsx_storeu_f32(dout, lsx_mul_f32(vres0, vsum0));
    din += 4;
    dout += 4;
  }
  for (int i = 0; i < rem_4; i++) {
    dout[0] =
        std::min(std::max(0.f, din[0] + offset), threshold) * din[0] / scale;
    dout++;
    din++;
  }
}

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
