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

#include "lite/backends/arm/math/fill_bias_relu.h"
#include <algorithm>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void fill_bias_relu<float>(float* tensor,
                           const float* bias,
                           int channel,
                           int channel_size,
                           bool flag_bias,
                           bool flag_relu) {
  float* data = tensor;
  if (flag_relu) {
    for (int j = 0; j < channel; ++j) {
      float bias_data = flag_bias ? bias[j] : 0.f;
      float32x4_t vbias = vdupq_n_f32(bias_data);
      float32x4_t vzero = vdupq_n_f32(0.f);
      int i = 0;
      for (; i < channel_size - 3; i += 4) {
        float32x4_t vdata = vld1q_f32(&data[i]);
        vdata = vaddq_f32(vdata, vbias);
        float32x4_t vmax = vmaxq_f32(vdata, vzero);
        vst1q_f32(data + i, vmax);
      }
      for (; i < channel_size; i++) {
        data[i] += bias_data;
        data[i] = data[i] > 0 ? data[i] : 0.f;
      }
      data += channel_size;
    }
  } else {
    for (int j = 0; j < channel; ++j) {
      float bias_data = flag_bias ? bias[j] : 0.f;
      float32x4_t vbias = vdupq_n_f32(bias_data);
      int i = 0;
      for (; i < channel_size - 3; i += 4) {
        float32x4_t vdata = vld1q_f32(&data[i]);
        vdata = vaddq_f32(vdata, vbias);
        vst1q_f32(data + i, vdata);
      }
      for (; i < channel_size; i++) {
        data[i] += bias_data;
      }
      data += channel_size;
    }
  }
}

template <>
void fill_bias_relu<int>(int* tensor,
                         const int* bias,
                         int channel,
                         int channel_size,
                         bool flag_bias,
                         bool flag_relu) {
  int* data = tensor;
  if (flag_relu) {
    for (int j = 0; j < channel; ++j) {
      int bias_data = flag_bias ? bias[j] : 0;
      int32x4_t vbias = vdupq_n_s32(bias_data);
      int32x4_t vzero = vdupq_n_s32(0);
      int i = 0;
      for (; i < channel_size - 7; i += 8) {
        int32x4_t vdata1 = vld1q_s32(data + i);
        int32x4_t vdata2 = vld1q_s32(data + i + 4);
        vdata1 = vaddq_s32(vdata1, vbias);
        vdata2 = vaddq_s32(vdata2, vbias);
        int32x4_t vmax1 = vmaxq_s32(vdata1, vzero);
        int32x4_t vmax2 = vmaxq_s32(vdata2, vzero);
        vst1q_s32(data + i, vmax1);
        vst1q_s32(data + i + 4, vmax2);
      }
      for (; i < channel_size; i++) {
        data[i] += bias_data;
        data[i] = data[i] > 0 ? data[i] : 0;
      }
      data += channel_size;
    }
  } else {
    for (int j = 0; j < channel; ++j) {
      int bias_data = flag_bias ? bias[j] : 0;
      int32x4_t vbias = vdupq_n_s32(bias_data);
      int i = 0;
      for (; i < channel_size - 7; i += 8) {
        int32x4_t vdata1 = vld1q_s32(data + i);
        int32x4_t vdata2 = vld1q_s32(data + i + 4);
        vdata1 = vaddq_s32(vdata1, vbias);
        vdata2 = vaddq_s32(vdata2, vbias);
        vst1q_s32(data + i, vdata1);
        vst1q_s32(data + i + 4, vdata2);
      }
      for (; i < channel_size; i++) {
        data[i] += bias_data;
      }
      data += channel_size;
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
