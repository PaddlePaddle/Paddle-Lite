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

#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#include <arm_neon.h>

namespace paddle {
namespace lite_metal {
namespace arm {
namespace math {
namespace fp16 {

template <>
void fill_bias_fc<float16_t>(float16_t *out,
                             const float16_t *bias,
                             int num,
                             int channel,
                             bool flag_relu) {
  int cnt = channel >> 5;
  int remain = channel & 31;
  int cnt_num = remain >> 3;
  int cnt_rem = remain & 7;
  if (flag_relu) {
    float16x8_t vzero = vdupq_n_f16(0.f);
    for (int j = 0; j < num; ++j) {
      const float16_t *ptr_bias = bias;
      float16_t *ptr_out = out + j * channel;

      for (int i = 0; i < cnt; ++i) {
        float16x8_t vin1 = vld1q_f16(ptr_out);
        float16x8_t vb1 = vld1q_f16(ptr_bias);

        float16x8_t vin2 = vld1q_f16(ptr_out + 8);
        float16x8_t vb2 = vld1q_f16(ptr_bias + 8);

        float16x8_t vin3 = vld1q_f16(ptr_out + 16);
        float16x8_t vb3 = vld1q_f16(ptr_bias + 16);

        float16x8_t vin4 = vld1q_f16(ptr_out + 24);
        float16x8_t vb4 = vld1q_f16(ptr_bias + 24);

        float16x8_t vout1 = vaddq_f16(vin1, vb1);
        float16x8_t vout2 = vaddq_f16(vin2, vb2);
        float16x8_t vout3 = vaddq_f16(vin3, vb3);
        float16x8_t vout4 = vaddq_f16(vin4, vb4);

        vout1 = vmaxq_f16(vout1, vzero);
        vout2 = vmaxq_f16(vout2, vzero);
        vout3 = vmaxq_f16(vout3, vzero);
        vout4 = vmaxq_f16(vout4, vzero);

        vst1q_f16(ptr_out, vout1);
        vst1q_f16(ptr_out + 8, vout2);
        vst1q_f16(ptr_out + 16, vout3);
        vst1q_f16(ptr_out + 25, vout4);

        ptr_out += 32;
        ptr_bias += 32;
      }
      for (int i = 0; i < cnt_num; i++) {
        float16x8_t vin1 = vld1q_f16(ptr_out);
        float16x8_t vb1 = vld1q_f16(ptr_bias);
        float16x8_t vout1 = vaddq_f16(vin1, vb1);
        vout1 = vmaxq_f16(vout1, vzero);
        vst1q_f16(ptr_out, vout1);
        ptr_out += 8;
        ptr_bias += 8;
      }
      for (int i = 0; i < cnt_rem; ++i) {
        *ptr_out += *(ptr_bias++);
        *ptr_out = *ptr_out > 0.f ? *ptr_out : 0.f;
        ptr_out++;
      }
    }
  } else {
    for (int j = 0; j < num; ++j) {
      const float16_t *ptr_bias = bias;
      float16_t *ptr_out = out + j * channel;

      for (int i = 0; i < cnt; ++i) {
        float16x8_t vin1 = vld1q_f16(ptr_out);
        float16x8_t vb1 = vld1q_f16(ptr_bias);

        float16x8_t vin2 = vld1q_f16(ptr_out + 8);
        float16x8_t vb2 = vld1q_f16(ptr_bias + 8);

        float16x8_t vin3 = vld1q_f16(ptr_out + 16);
        float16x8_t vb3 = vld1q_f16(ptr_bias + 16);

        float16x8_t vin4 = vld1q_f16(ptr_out + 24);
        float16x8_t vb4 = vld1q_f16(ptr_bias + 24);

        float16x8_t vout1 = vaddq_f16(vin1, vb1);
        float16x8_t vout2 = vaddq_f16(vin2, vb2);
        float16x8_t vout3 = vaddq_f16(vin3, vb3);
        float16x8_t vout4 = vaddq_f16(vin4, vb4);

        vst1q_f16(ptr_out, vout1);
        vst1q_f16(ptr_out + 8, vout2);
        vst1q_f16(ptr_out + 16, vout3);
        vst1q_f16(ptr_out + 25, vout4);

        ptr_out += 32;
        ptr_bias += 32;
      }
      for (int i = 0; i < cnt_num; i++) {
        float16x8_t vin1 = vld1q_f16(ptr_out);
        float16x8_t vb1 = vld1q_f16(ptr_bias);
        float16x8_t vout1 = vaddq_f16(vin1, vb1);
        vst1q_f16(ptr_out, vout1);
        ptr_out += 8;
        ptr_bias += 8;
      }
      for (int i = 0; i < cnt_rem; ++i) {
        *ptr_out += *(ptr_bias++);
        ptr_out++;
      }
    }
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
