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

#include "lite/backends/arm/math/clip.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/saturate.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void clip_kernel_fp32(
    const float* input, int64_t num, float min, float max, float* output) {
  const float* din_ptr = input;
  float* dout_ptr = output;
  int64_t cnt = num >> 4;
  int remain = num % 16;
  float32x4_t max_val = vdupq_n_f32(max);
  float32x4_t min_val = vdupq_n_f32(min);
  for (int64_t n = 0; n < cnt; n++) {
    float32x4_t tmp0 =
        vminq_f32(vmaxq_f32(vld1q_f32(din_ptr), min_val), max_val);
    float32x4_t tmp1 =
        vminq_f32(vmaxq_f32(vld1q_f32(din_ptr + 4), min_val), max_val);
    float32x4_t tmp2 =
        vminq_f32(vmaxq_f32(vld1q_f32(din_ptr + 8), min_val), max_val);
    float32x4_t tmp3 =
        vminq_f32(vmaxq_f32(vld1q_f32(din_ptr + 12), min_val), max_val);
    vst1q_f32(dout_ptr, tmp0);
    vst1q_f32(dout_ptr + 4, tmp1);
    vst1q_f32(dout_ptr + 8, tmp2);
    vst1q_f32(dout_ptr + 12, tmp3);
    din_ptr += 16;
    dout_ptr += 16;
  }
  for (int i = 0; i < remain; i++) {
    float tmp = din_ptr[i] > min ? din_ptr[i] : min;
    dout_ptr[i] = tmp < max ? tmp : max;
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
