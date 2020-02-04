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

#include "lite/backends/arm/math/funcs.h"
#include <arm_neon.h>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void fill_bias_fc<float>(
    float *out, const float *bias, int num, int channel, bool flag_relu) {
  int cnt = channel >> 4;
  int remain = channel & 15;
  if (flag_relu) {
    float32x4_t vzero = vdupq_n_f32(0.f);
    for (int j = 0; j < num; ++j) {
      const float *ptr_bias = bias;
      float *ptr_out = out + j * channel;

      for (int i = 0; i < cnt; ++i) {
        float32x4_t vin1 = vld1q_f32(ptr_out);
        float32x4_t vb1 = vld1q_f32(ptr_bias);

        float32x4_t vin2 = vld1q_f32(ptr_out + 4);
        float32x4_t vb2 = vld1q_f32(ptr_bias + 4);

        float32x4_t vin3 = vld1q_f32(ptr_out + 8);
        float32x4_t vb3 = vld1q_f32(ptr_bias + 8);

        float32x4_t vin4 = vld1q_f32(ptr_out + 12);
        float32x4_t vb4 = vld1q_f32(ptr_bias + 12);

        float32x4_t vout1 = vaddq_f32(vin1, vb1);
        float32x4_t vout2 = vaddq_f32(vin2, vb2);
        float32x4_t vout3 = vaddq_f32(vin3, vb3);
        float32x4_t vout4 = vaddq_f32(vin4, vb4);

        vout1 = vmaxq_f32(vout1, vzero);
        vout2 = vmaxq_f32(vout2, vzero);
        vout3 = vmaxq_f32(vout3, vzero);
        vout4 = vmaxq_f32(vout4, vzero);

        vst1q_f32(ptr_out, vout1);
        vst1q_f32(ptr_out + 4, vout2);
        vst1q_f32(ptr_out + 8, vout3);
        vst1q_f32(ptr_out + 12, vout4);

        ptr_out += 16;
        ptr_bias += 16;
      }
      for (int i = 0; i < remain; ++i) {
        *ptr_out += *(ptr_bias++);
        *ptr_out = *ptr_out > 0.f ? *ptr_out : 0.f;
        ptr_out++;
      }
    }
  } else {
    for (int j = 0; j < num; ++j) {
      const float *ptr_bias = bias;
      float *ptr_out = out + j * channel;

      for (int i = 0; i < cnt; ++i) {
        float32x4_t vin1 = vld1q_f32(ptr_out);
        float32x4_t vb1 = vld1q_f32(ptr_bias);

        float32x4_t vin2 = vld1q_f32(ptr_out + 4);
        float32x4_t vb2 = vld1q_f32(ptr_bias + 4);

        float32x4_t vin3 = vld1q_f32(ptr_out + 8);
        float32x4_t vb3 = vld1q_f32(ptr_bias + 8);

        float32x4_t vin4 = vld1q_f32(ptr_out + 12);
        float32x4_t vb4 = vld1q_f32(ptr_bias + 12);

        float32x4_t vout1 = vaddq_f32(vin1, vb1);
        float32x4_t vout2 = vaddq_f32(vin2, vb2);
        float32x4_t vout3 = vaddq_f32(vin3, vb3);
        float32x4_t vout4 = vaddq_f32(vin4, vb4);

        vst1q_f32(ptr_out, vout1);
        vst1q_f32(ptr_out + 4, vout2);
        vst1q_f32(ptr_out + 8, vout3);
        vst1q_f32(ptr_out + 12, vout4);

        ptr_out += 16;
        ptr_bias += 16;
      }
      for (int i = 0; i < remain; ++i) {
        *(ptr_out++) += *(ptr_bias++);
      }
    }
  }
}

template <>
void fill_bias_fc<int>(
    int *out, const int *bias, int num, int channel, bool flag_relu) {
  int cnt = channel >> 4;
  int remain = channel & 15;
  if (flag_relu) {
    for (int j = 0; j < num; ++j) {
      const int *ptr_bias = bias;
      int *ptr_out = out + j * channel;

      int32x4_t vzero = vdupq_n_s32(0);

      for (int i = 0; i < cnt; ++i) {
        int32x4_t vin1 = vld1q_s32(ptr_out);
        int32x4_t vb1 = vld1q_s32(ptr_bias);

        int32x4_t vin2 = vld1q_s32(ptr_out + 4);
        int32x4_t vb2 = vld1q_s32(ptr_bias + 4);

        int32x4_t vin3 = vld1q_s32(ptr_out + 8);
        int32x4_t vb3 = vld1q_s32(ptr_bias + 8);

        int32x4_t vin4 = vld1q_s32(ptr_out + 12);
        int32x4_t vb4 = vld1q_s32(ptr_bias + 12);

        int32x4_t vout1 = vaddq_s32(vin1, vb1);
        int32x4_t vout2 = vaddq_s32(vin2, vb2);
        int32x4_t vout3 = vaddq_s32(vin3, vb3);
        int32x4_t vout4 = vaddq_s32(vin4, vb4);

        vout1 = vmaxq_s32(vout1, vzero);
        vout2 = vmaxq_s32(vout2, vzero);
        vout3 = vmaxq_s32(vout3, vzero);
        vout4 = vmaxq_s32(vout4, vzero);

        vst1q_s32(ptr_out, vout1);
        vst1q_s32(ptr_out + 4, vout2);
        vst1q_s32(ptr_out + 8, vout3);
        vst1q_s32(ptr_out + 12, vout4);

        ptr_out += 16;
        ptr_bias += 16;
      }
      for (int i = 0; i < remain; ++i) {
        *ptr_out += *(ptr_bias++);
        *ptr_out = *ptr_out > 0 ? *ptr_out : 0;
        ptr_out++;
      }
    }
  } else {
    for (int j = 0; j < num; ++j) {
      const int *ptr_bias = bias;
      int *ptr_out = out + j * channel;

      int32x4_t vout1;
      int32x4_t vout2;
      int32x4_t vout3;
      int32x4_t vout4;

      for (int i = 0; i < cnt; ++i) {
        int32x4_t vin1 = vld1q_s32(ptr_out);
        int32x4_t vb1 = vld1q_s32(ptr_bias);

        int32x4_t vin2 = vld1q_s32(ptr_out + 4);
        int32x4_t vb2 = vld1q_s32(ptr_bias + 4);

        int32x4_t vin3 = vld1q_s32(ptr_out + 8);
        int32x4_t vb3 = vld1q_s32(ptr_bias + 8);

        int32x4_t vin4 = vld1q_s32(ptr_out + 12);
        int32x4_t vb4 = vld1q_s32(ptr_bias + 12);

        vout1 = vaddq_s32(vin1, vb1);
        vout2 = vaddq_s32(vin2, vb2);
        vout3 = vaddq_s32(vin3, vb3);
        vout4 = vaddq_s32(vin4, vb4);

        vst1q_s32(ptr_out, vout1);
        vst1q_s32(ptr_out + 4, vout2);
        vst1q_s32(ptr_out + 8, vout3);
        vst1q_s32(ptr_out + 12, vout4);

        ptr_out += 16;
        ptr_bias += 16;
      }
      for (int i = 0; i < remain; ++i) {
        *(ptr_out++) += *(ptr_bias++);
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
