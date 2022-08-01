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

#include "lite/backends/arm/math/power.h"

#include "lite/backends/arm/math/funcs.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
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
  if (fabsf(factor_ - 1.f) < 1e-6f) {
    _do_power = false;
  }
  if (fabsf(scale_ - 1.f) < 1e-6f) {
    _do_scale = false;
  }
  if (fabsf(shift_ - 0.f) < 1e-6f) {
    _do_shift = false;
  }
  float* ptr_out = dout;
  const float* ptr_in = din;
  float32x4_t vscale = vdupq_n_f32(scale_);
  float32x4_t vshift = vdupq_n_f32(shift_);
  float32x4_t vpower = vdupq_n_f32(factor_);
  LITE_PARALLEL_BEGIN(nums, tid, cnt) {
    float32x4_t vr0 = vld1q_f32(ptr_in);
    ptr_in += 4;
    float32x4_t vr1 = vld1q_f32(ptr_in);
    ptr_in += 4;
    float32x4_t vr2 = vld1q_f32(ptr_in);
    ptr_in += 4;
    float32x4_t vr3 = vld1q_f32(ptr_in);
    ptr_in += 4;
    if (_do_scale) {
      vr0 = vmulq_f32(vr0, vscale);
      vr1 = vmulq_f32(vr1, vscale);
      vr2 = vmulq_f32(vr2, vscale);
      vr3 = vmulq_f32(vr3, vscale);
    }
    if (_do_shift) {
      vr0 = vaddq_f32(vr0, vshift);
      vr1 = vaddq_f32(vr1, vshift);
      vr2 = vaddq_f32(vr2, vshift);
      vr3 = vaddq_f32(vr3, vshift);
    }
    if (_do_power) {
      vr0 = pow_ps(vr0, vpower);
      vr1 = pow_ps(vr1, vpower);
      vr2 = pow_ps(vr2, vpower);
      vr3 = pow_ps(vr3, vpower);
    }
    vst1q_f32(ptr_out, vr0);
    ptr_out += 4;
    vst1q_f32(ptr_out, vr1);
    ptr_out += 4;
    vst1q_f32(ptr_out, vr2);
    ptr_out += 4;
    vst1q_f32(ptr_out, vr3);
    ptr_out += 4;
  }
  LITE_PARALLEL_END()
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = std::pow((ptr_in[0] * scale_ + shift_), factor_);
    ptr_in++;
    ptr_out++;
  }
}

} /* namespace math */
} /* namespace arm */
} /* namespace lite */
} /* namespace paddle */
