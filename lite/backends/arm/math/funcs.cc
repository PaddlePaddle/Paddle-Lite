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

#define LOADA_DATA_16                         \
  float32x4_t vin1 = vld1q_f32(ptr_out);      \
  float32x4_t vb1 = vld1q_f32(ptr_bias);      \
  float32x4_t vin2 = vld1q_f32(ptr_out + 4);  \
  float32x4_t vb2 = vld1q_f32(ptr_bias + 4);  \
  float32x4_t vin3 = vld1q_f32(ptr_out + 8);  \
  float32x4_t vb3 = vld1q_f32(ptr_bias + 8);  \
  float32x4_t vin4 = vld1q_f32(ptr_out + 12); \
  float32x4_t vb4 = vld1q_f32(ptr_bias + 12); \
  float32x4_t vout1 = vaddq_f32(vin1, vb1);   \
  float32x4_t vout2 = vaddq_f32(vin2, vb2);   \
  float32x4_t vout3 = vaddq_f32(vin3, vb3);   \
  float32x4_t vout4 = vaddq_f32(vin4, vb4);
#define RELU_16                    \
  vout1 = vmaxq_f32(vout1, vzero); \
  vout2 = vmaxq_f32(vout2, vzero); \
  vout3 = vmaxq_f32(vout3, vzero); \
  vout4 = vmaxq_f32(vout4, vzero);
#define RELU6_16                    \
  vout1 = vminq_f32(vout1, valpha); \
  vout2 = vminq_f32(vout2, valpha); \
  vout3 = vminq_f32(vout3, valpha); \
  vout4 = vminq_f32(vout4, valpha);
#define STORE_16                  \
  vst1q_f32(ptr_out, vout1);      \
  vst1q_f32(ptr_out + 4, vout2);  \
  vst1q_f32(ptr_out + 8, vout3);  \
  vst1q_f32(ptr_out + 12, vout4); \
  ptr_out += 16;                  \
  ptr_bias += 16;
#define LOADA_DATA_4                     \
  float32x4_t vin1 = vld1q_f32(ptr_out); \
  float32x4_t vb1 = vld1q_f32(ptr_bias); \
  float32x4_t vout1 = vaddq_f32(vin1, vb1);
#define RELU_4 vout1 = vmaxq_f32(vout1, vzero);
#define RELU6_4 vout1 = vminq_f32(vout1, valpha);
#define STORE_4              \
  vst1q_f32(ptr_out, vout1); \
  ptr_out += 4;              \
  ptr_bias += 4;

template <>
void fill_bias_fc(float *out,
                  const float *bias,
                  int num,
                  int channel,
                  const operators::ActivationParam *act_param) {
  int cnt = channel >> 4;
  int remain = channel & 15;
  int cnt_num = remain >> 2;
  int cnt_rem = remain & 3;

  if (act_param != nullptr && act_param->has_active) {
    float32x4_t vzero = vdupq_n_f32(0.f);
    if (act_param->active_type == lite_api::ActivationType::kRelu) {
      for (int j = 0; j < num; ++j) {
        const float *ptr_bias = bias;
        float *ptr_out = out + j * channel;
        for (int i = 0; i < cnt; ++i) {
          LOADA_DATA_16
          RELU_16
          STORE_16
        }
        for (int i = 0; i < cnt_num; ++i) {
          LOADA_DATA_4
          RELU_4
          STORE_4
        }
        for (int i = 0; i < cnt_rem; ++i) {
          *ptr_out += *(ptr_bias++);
          *ptr_out = *ptr_out > 0.f ? *ptr_out : 0.f;
          ptr_out++;
        }
      }
    } else if (act_param->active_type == lite_api::ActivationType::kRelu6) {
      float alpha = act_param->Relu_clipped_coef;
      float32x4_t valpha = vdupq_n_f32(act_param->Relu_clipped_coef);
      for (int j = 0; j < num; ++j) {
        const float *ptr_bias = bias;
        float *ptr_out = out + j * channel;
        for (int i = 0; i < cnt; ++i) {
          LOADA_DATA_16
          RELU_16
          RELU6_16
          STORE_16
        }
        for (int i = 0; i < cnt_num; ++i) {
          LOADA_DATA_4
          RELU_4
          RELU6_4
          STORE_4
        }
        for (int i = 0; i < cnt_rem; ++i) {
          *ptr_out += *(ptr_bias++);
          *ptr_out =
              *ptr_out > 0.f ? ((*ptr_out < alpha) ? *ptr_out : alpha) : 0.f;
          ptr_out++;
        }
      }
    } else {
      LOG(FATAL) << "This act_type: "
                 << static_cast<int>(act_param->active_type)
                 << " doesn't support";
    }
  } else {
    for (int j = 0; j < num; ++j) {
      const float *ptr_bias = bias;
      float *ptr_out = out + j * channel;

      for (int i = 0; i < cnt; ++i) {
        LOADA_DATA_16
        STORE_16
      }
      for (int i = 0; i < cnt_num; ++i) {
        LOADA_DATA_4
        STORE_4
      }
      for (int i = 0; i < cnt_rem; ++i) {
        *(ptr_out++) += *(ptr_bias++);
      }
    }
  }
}
#undef LOADA_DATA_16
#undef RELU_16
#undef RELU6_16
#undef STORE_16
#undef LOADA_DATA_4
#undef RELU_4
#undef RELU6_4
#undef STORE_4
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
