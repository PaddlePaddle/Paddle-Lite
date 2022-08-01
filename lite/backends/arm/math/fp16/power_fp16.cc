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

#include "lite/backends/arm/math/fp16/power_fp16.h"
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

void power_fp16(const float16_t* din,
                float16_t* dout,
                const int num,
                float scale_,
                float shift_,
                float factor_) {
  int cnt = num >> 5;
  int remain = num % 32;
  bool _do_power = true;
  bool _do_scale = true;
  bool _do_shift = true;
  int rem_cnt = remain >> 3;
  int rem_rem = remain % 8;
  if (fabsf(factor_ - 1.f) < 1e-6f) {
    _do_power = false;
  }
  if (fabsf(scale_ - 1.f) < 1e-6f) {
    _do_scale = false;
  }
  if (fabsf(shift_ - 0.f) < 1e-6f) {
    _do_shift = false;
  }
  float16_t* ptr_out = dout;
  const float16_t* ptr_in = din;
  // TODO: ndk22 bug,
  // when use vget_low_f16 will occur compilation bug in this code,
  // need study better way to fix this question
  float16_t scale_tmp[8];
  float16_t shift_tmp[8];
  for (int i = 0; i < 8; i++) {
    scale_tmp[i] = static_cast<float16_t>(scale_);
    shift_tmp[i] = static_cast<float16_t>(shift_);
  }
  float16x8_t vscale = vld1q_f16(scale_tmp);
  float16x8_t vshift = vld1q_f16(shift_tmp);
  float32x4_t vpower = vdupq_n_f32(factor_);
  LITE_PARALLEL_BEGIN(nums, tid, cnt) {
    float16x8_t vr0 = vld1q_f16(ptr_in);
    float16x8_t vr1 = vld1q_f16(ptr_in + 8);
    float16x8_t vr2 = vld1q_f16(ptr_in + 16);
    float16x8_t vr3 = vld1q_f16(ptr_in + 24);
    ptr_in += 32;
    if (_do_scale) {
      vr0 = vmulq_f16(vr0, vscale);
      vr1 = vmulq_f16(vr1, vscale);
      vr2 = vmulq_f16(vr2, vscale);
      vr3 = vmulq_f16(vr3, vscale);
    }
    if (_do_shift) {
      vr0 = vaddq_f16(vr0, vshift);
      vr1 = vaddq_f16(vr1, vshift);
      vr2 = vaddq_f16(vr2, vshift);
      vr3 = vaddq_f16(vr3, vshift);
    }
    if (_do_power) {
      vr0 = powq_ps_f16(vr0, vpower);
      vr1 = powq_ps_f16(vr1, vpower);
      vr2 = powq_ps_f16(vr2, vpower);
      vr3 = powq_ps_f16(vr3, vpower);
    }
    vst1q_f16(ptr_out, vr0);
    vst1q_f16(ptr_out + 8, vr1);
    vst1q_f16(ptr_out + 16, vr2);
    vst1q_f16(ptr_out + 24, vr3);
    ptr_out += 32;
  }
  LITE_PARALLEL_END()
  LITE_PARALLEL_BEGIN(nums, tid, rem_cnt) {
    float16x8_t vr0 = vld1q_f16(ptr_in);
    ptr_in += 8;
    if (_do_scale) {
      vr0 = vmulq_f16(vr0, vscale);
    }
    if (_do_shift) {
      vr0 = vaddq_f16(vr0, vshift);
    }
    if (_do_power) {
      vr0 = powq_ps_f16(vr0, vpower);
    }
    vst1q_f16(ptr_out, vr0);
    ptr_out += 8;
  }
  LITE_PARALLEL_END()
  if (rem_rem > 4) {
    rem_rem -= 4;
    float16x4_t vr0 = vld1_f16(ptr_in);
    ptr_in += 4;
    if (_do_scale) {
      vr0 = vmul_f16(vr0, vget_low_f16(vscale));
    }
    if (_do_shift) {
      vr0 = vadd_f16(vr0, vget_low_f16(vshift));
    }
    if (_do_power) {
      vr0 = pow_ps_f16(vr0, vpower);
    }
    vst1_f16(ptr_out, vr0);
    ptr_out += 4;
  }
  for (int j = 0; j < rem_rem; ++j) {
    ptr_out[0] = std::pow((ptr_in[0] * (float16_t)scale_ + (float16_t)shift_),
                          (float16_t)factor_);
    ptr_in++;
    ptr_out++;
  }
}

} /* namespace fp16 */
} /* namespace math */
} /* namespace arm */
} /* namespace lite */
} /* namespace paddle */
