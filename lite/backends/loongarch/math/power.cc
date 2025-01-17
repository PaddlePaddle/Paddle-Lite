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

#include "lite/backends/loongarch/math/power.h"
#include "lite/backends/loongarch/xxl.h"
#include <cmath>
#include "lite/backends/loongarch/math/include/mathfuns.h"

namespace paddle {
namespace lite {
namespace loongarch {
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
#ifdef __loongarch_asx
  __m256 vscale_256 = lasx_set1_f32(scale_);
  __m256 vshift_256 = lasx_set1_f32(shift_);
  __m256 vfactor_256 = lasx_set1_f32(factor_);
#endif
  __m128 vscale = lsx_set1_f32(scale_);
  __m128 vshift = lsx_set1_f32(shift_);
  float* ptr_out = dout;
  const float* ptr_in = din;
  if (_do_power) {
    for (int i = 0; i < cnt; i++) {
#ifdef __loongarch_asx
      __m256 vin0 = lasx_loadu_f32(ptr_in);
      __m256 vin1 = lasx_loadu_f32(ptr_in + 8);
      ptr_in += 16;
      __m256 vsum0 = lasx_mul_f32(vin0, vscale_256);
      __m256 vsum1 = lasx_mul_f32(vin1, vscale_256);
      __m256 vres0 = lasx_add_f32(vsum0, vshift_256);
      __m256 vres1 = lasx_add_f32(vsum1, vshift_256);
      vres0 = pow256_ps(vres0, vfactor_256);
      vres1 = pow256_ps(vres1, vfactor_256);
      lasx_storeu_f32(ptr_out, vres0);
      lasx_storeu_f32(ptr_out + 8, vres1);
#else
      __m128 vin0 = lsx_loadu_f32(ptr_in);
      __m128 vin1 = lsx_loadu_f32(ptr_in + 4);
      __m128 vin2 = lsx_loadu_f32(ptr_in + 8);
      __m128 vin3 = lsx_loadu_f32(ptr_in + 12);
      __m128 vsum0 = lsx_mul_f32(vin0, vscale);
      __m128 vsum1 = lsx_mul_f32(vin1, vscale);
      __m128 vsum2 = lsx_mul_f32(vin2, vscale);
      __m128 vsum3 = lsx_mul_f32(vin3, vscale);
      __m128 vres0 = lsx_add_f32(vsum0, vshift);
      __m128 vres1 = lsx_add_f32(vsum1, vshift);
      __m128 vres2 = lsx_add_f32(vsum2, vshift);
      __m128 vres3 = lsx_add_f32(vsum3, vshift);

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
      __m128 vin0 = lsx_loadu_f32(ptr_in);
      ptr_in += 4;
      __m128 vsum0 = lsx_mul_f32(vin0, vscale);
      __m128 vres0 = lsx_add_f32(vsum0, vshift);
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
#ifdef __loongarch_asx
      __m256 vin0 = lasx_loadu_f32(ptr_in);
      __m256 vin1 = lasx_loadu_f32(ptr_in + 8);
      ptr_in += 16;
      __m256 vsum0 = lasx_mul_f32(vin0, vscale_256);
      __m256 vsum1 = lasx_mul_f32(vin1, vscale_256);
      __m256 vres0 = lasx_add_f32(vsum0, vshift_256);
      __m256 vres1 = lasx_add_f32(vsum1, vshift_256);
      lasx_storeu_f32(ptr_out, vres0);
      lasx_storeu_f32(ptr_out + 8, vres1);
      ptr_out += 16;
#else
      __m128 vin0 = lsx_loadu_f32(ptr_in);
      __m128 vin1 = lsx_loadu_f32(ptr_in + 4);
      __m128 vin2 = lsx_loadu_f32(ptr_in + 8);
      __m128 vin3 = lsx_loadu_f32(ptr_in + 12);
      __m128 vsum0 = lsx_mul_f32(vin0, vscale);
      __m128 vsum1 = lsx_mul_f32(vin1, vscale);
      __m128 vsum2 = lsx_mul_f32(vin2, vscale);
      __m128 vsum3 = lsx_mul_f32(vin3, vscale);
      __m128 vres0 = lsx_add_f32(vsum0, vshift);
      __m128 vres1 = lsx_add_f32(vsum1, vshift);
      __m128 vres2 = lsx_add_f32(vsum2, vshift);
      __m128 vres3 = lsx_add_f32(vsum3, vshift);

      ptr_in += 16;
      lsx_storeu_f32(ptr_out, vres0);
      lsx_storeu_f32(ptr_out + 4, vres1);
      lsx_storeu_f32(ptr_out + 8, vres2);
      lsx_storeu_f32(ptr_out + 12, vres3);
      ptr_out += 16;
#endif
    }
    for (int i = 0; i < rem_cnt; i++) {
      __m128 vin0 = lsx_loadu_f32(ptr_in);
      ptr_in += 4;
      __m128 vsum0 = lsx_mul_f32(vin0, vscale);
      __m128 vres0 = lsx_add_f32(vsum0, vshift);

      lsx_storeu_f32(ptr_out, vres0);
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
} /* namespace loongarch */
} /* namespace lite */
} /* namespace paddle */
