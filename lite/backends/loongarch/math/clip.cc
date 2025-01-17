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

#include "lite/backends/loongarch/math/clip.h"
#include "lite/backends/loongarch/xxl.h"

namespace paddle {
namespace lite {
namespace loongarch {
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
#ifdef __loongarch_asx
  __m256 max_256 = lasx_set1_f32(max_);
  __m256 min_256 = lasx_set1_f32(min_);
#endif
  __m128 vmax = lsx_set1_f32(max_);
  __m128 vmin = lsx_set1_f32(min_);
  for (int i = 0; i < cnt; i++) {
#ifdef __loongarch_asx
    __m256 vin0 = lasx_loadu_f32(ptr_in);
    __m256 vin1 = lasx_loadu_f32(ptr_in + 8);
    vin0 = lasx_min_f32(lasx_max_f32(vin0, min_256), max_256);
    vin1 = lasx_min_f32(lasx_max_f32(vin1, min_256), max_256);
    lasx_storeu_f32(ptr_out, vin0);
    lasx_storeu_f32(ptr_out + 8, vin1);
#else
    __m128 vin0 = lsx_loadu_f32(ptr_in);
    __m128 vin1 = lsx_loadu_f32(ptr_in + 4);
    __m128 vin2 = lsx_loadu_f32(ptr_in + 8);
    __m128 vin3 = lsx_loadu_f32(ptr_in + 12);

    vin0 = lsx_min_f32(lsx_max_f32(vin0, vmin), vmax);
    vin1 = lsx_min_f32(lsx_max_f32(vin1, vmin), vmax);
    vin2 = lsx_min_f32(lsx_max_f32(vin2, vmin), vmax);
    vin3 = lsx_min_f32(lsx_max_f32(vin3, vmin), vmax);

    lsx_storeu_f32(ptr_out, vin0);
    lsx_storeu_f32(ptr_out + 4, vin1);
    lsx_storeu_f32(ptr_out + 8, vin2);
    lsx_storeu_f32(ptr_out + 12, vin3);
#endif
    ptr_in += 16;
    ptr_out += 16;
  }
  for (int i = 0; i < rem_cnt; i++) {
    __m128 vin0 = lsx_loadu_f32(ptr_in);
    vin0 = lsx_min_f32(lsx_max_f32(vin0, vmin), vmax);
    lsx_storeu_f32(ptr_out, vin0);
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
} /* namespace loongarch */
} /* namespace lite */
} /* namespace paddle */
