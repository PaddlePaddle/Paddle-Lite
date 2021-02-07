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

#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include "lite/backends/arm/math/fp16/conv3x3_depthwise_fp16.h"
#include "lite/backends/arm/math/fp16/conv_block_utils_fp16.h"
#include "lite/backends/arm/math/fp16/conv_impl_fp16.h"
#include "lite/backends/arm/math/fp16/gemm_fp16.h"
#include "lite/backends/arm/math/fp16/softmax_fp16.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4
#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

// exp() computed for 8 float at once
inline float16x8_t expq_ps_f16(float16x8_t x) {
  float16x8_t one = vdupq_n_f16(1.f);
  x = vminq_f16(x, vdupq_n_f16(c_exp_hi));
  x = vmaxq_f16(x, vdupq_n_f16(c_exp_lo));

  // express exp(x) as exp(g + n*log(2))
  float16x8_t fx =
      vmlaq_f16(vdupq_n_f16(0.5f), x, vdupq_n_f16(c_cephes_LOG2EF));

  // perform a floorf
  float16x8_t tmp = vcvtq_f16_s16(vcvtq_s16_f16(fx));

  // if greater, substract 1
  uint16x8_t mask = vcgtq_f16(tmp, fx);
  mask = vandq_u16(mask, vreinterpretq_u16_f16(one));

  fx = vsubq_f16(tmp, vreinterpretq_f16_u16(mask));

  tmp = vmulq_f16(fx, vdupq_n_f16(c_cephes_exp_C1));
  float16x8_t z = vmulq_f16(fx, vdupq_n_f16(c_cephes_exp_C2));
  x = vsubq_f16(x, tmp);
  x = vsubq_f16(x, z);

  static const float16_t cephes_exp_p[6] = {c_cephes_exp_p0,
                                            c_cephes_exp_p1,
                                            c_cephes_exp_p2,
                                            c_cephes_exp_p3,
                                            c_cephes_exp_p4,
                                            c_cephes_exp_p5};
  float16x8_t y = vld1q_dup_f16(cephes_exp_p + 0);
  float16x8_t c1 = vld1q_dup_f16(cephes_exp_p + 1);
  float16x8_t c2 = vld1q_dup_f16(cephes_exp_p + 2);
  float16x8_t c3 = vld1q_dup_f16(cephes_exp_p + 3);
  float16x8_t c4 = vld1q_dup_f16(cephes_exp_p + 4);
  float16x8_t c5 = vld1q_dup_f16(cephes_exp_p + 5);

  y = vmulq_f16(y, x);
  z = vmulq_f16(x, x);

  y = vaddq_f16(y, c1);
  y = vmulq_f16(y, x);
  y = vaddq_f16(y, c2);
  y = vmulq_f16(y, x);
  y = vaddq_f16(y, c3);
  y = vmulq_f16(y, x);
  y = vaddq_f16(y, c4);
  y = vmulq_f16(y, x);
  y = vaddq_f16(y, c5);

  y = vmulq_f16(y, z);
  y = vaddq_f16(y, x);
  y = vaddq_f16(y, one);

  // build 2^n fp16=1+5+10
  int16x8_t mm = vcvtq_s16_f16(fx);

  mm = vaddq_s16(mm, vdupq_n_s16(0x1f));
  mm = vshlq_n_s16(mm, 10);
  float16x8_t pow2n = vreinterpretq_f16_s16(mm);
  y = vmulq_f16(y, pow2n);
  return y;
}

inline float16x4_t exp_ps_f16(float16x4_t x) {
  float16x4_t one = vdup_n_f16(1.f);
  x = vmin_f16(x, vdup_n_f16(c_exp_hi));
  x = vmax_f16(x, vdup_n_f16(c_exp_lo));

  // express exp(x) as exp(g + n*log(2))
  float16x4_t fx = vmla_f16(vdup_n_f16(0.5f), x, vdup_n_f16(c_cephes_LOG2EF));

  // perform a floorf
  float16x4_t tmp = vcvt_f16_s16(vcvt_s16_f16(fx));

  // if greater, substract 1
  uint16x4_t mask = vcgt_f16(tmp, fx);
  mask = vand_u16(mask, vreinterpret_u16_f16(one));

  fx = vsub_f16(tmp, vreinterpret_f16_u16(mask));

  tmp = vmul_f16(fx, vdup_n_f16(c_cephes_exp_C1));
  float16x4_t z = vmul_f16(fx, vdup_n_f16(c_cephes_exp_C2));
  x = vsub_f16(x, tmp);
  x = vsub_f16(x, z);

  static const float16_t cephes_exp_p[6] = {c_cephes_exp_p0,
                                            c_cephes_exp_p1,
                                            c_cephes_exp_p2,
                                            c_cephes_exp_p3,
                                            c_cephes_exp_p4,
                                            c_cephes_exp_p5};
  float16x4_t y = vld1_dup_f16(cephes_exp_p + 0);
  float16x4_t c1 = vld1_dup_f16(cephes_exp_p + 1);
  float16x4_t c2 = vld1_dup_f16(cephes_exp_p + 2);
  float16x4_t c3 = vld1_dup_f16(cephes_exp_p + 3);
  float16x4_t c4 = vld1_dup_f16(cephes_exp_p + 4);
  float16x4_t c5 = vld1_dup_f16(cephes_exp_p + 5);

  y = vmul_f16(y, x);
  z = vmul_f16(x, x);

  y = vadd_f16(y, c1);
  y = vmul_f16(y, x);
  y = vadd_f16(y, c2);
  y = vmul_f16(y, x);
  y = vadd_f16(y, c3);
  y = vmul_f16(y, x);
  y = vadd_f16(y, c4);
  y = vmul_f16(y, x);
  y = vadd_f16(y, c5);

  y = vmul_f16(y, z);
  y = vadd_f16(y, x);
  y = vadd_f16(y, one);

  // build 2^n fp16=1+5+10
  int16x4_t mm = vcvt_s16_f16(fx);

  mm = vadd_s16(mm, vdup_n_s16(0x1f));
  mm = vshl_n_s16(mm, 10);
  float16x4_t pow2n = vreinterpret_f16_s16(mm);
  y = vmul_f16(y, pow2n);
  return y;
}

inline float16x8_t divq_ps(float16x8_t a, float16x8_t b) {
  float16x8_t reciprocal = vrecpeq_f16(b);
  reciprocal = vmulq_f16(vrecpsq_f16(b, reciprocal), reciprocal);
  return vmulq_f16(a, reciprocal);
}

inline float16x4_t div_ps(float16x4_t a, float16x4_t b) {
  float16x4_t reciprocal = vrecpe_f16(b);
  reciprocal = vmul_f16(vrecps_f16(b, reciprocal), reciprocal);
  return vmul_f16(a, reciprocal);
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
