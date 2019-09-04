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

#include "lite/backends/arm/math/lrn.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void compute_across_channels<float>(const float* din,
                                    float* dout,
                                    int num,
                                    int channel,
                                    int h,
                                    int w,
                                    int local_size,
                                    float alpha,
                                    float beta,
                                    float k) {
  int channel_size = h * w;
  int cnt = channel_size / 4;
  int remain = channel_size % 4;
  int pre_pad = (local_size - 1) / 2;
  int post_pad = local_size - pre_pad - 1;
  float32x4_t k_val = vdupq_n_f32(k);
  float32x4_t alpha_val = vdupq_n_f32(alpha);
  float32x4_t beta_val = vdupq_n_f32(-beta);
  for (int n = 0; n < num; ++n) {
    const float* din_ptr = din + n * channel * channel_size;
    float* dout_ptr = dout + n * channel * channel_size;
    for (int c = 0; c < channel; ++c) {
      const float* din_ch_ptr = din_ptr + c * channel_size;
      float* dout_ch_ptr = dout_ptr + c * channel_size;
      int cs = (c - pre_pad) < 0 ? 0 : (c - pre_pad);
      int ce = (c + post_pad) >= channel ? channel : (c + pre_pad + 1);
      for (int i = 0; i < cnt; ++i) {
        int idx = i * 4;
        float32x4_t sum = vdupq_n_f32(0.f);
        float32x4_t din = vld1q_f32(din_ch_ptr);
        for (int k = cs; k < ce; ++k) {
          float32x4_t v0 = vld1q_f32(&din_ptr[k * channel_size + idx]);
          sum = vmlaq_f32(sum, v0, v0);
        }
        sum = vmulq_f32(sum, alpha_val);
        sum = vaddq_f32(sum, k_val);
        float32x4_t res0 = pow_ps(sum, beta_val);
        float32x4_t res1 = vmulq_f32(din, res0);
        vst1q_f32(dout_ch_ptr, res1);
        dout_ch_ptr += 4;
        din_ch_ptr += 4;
      }
      int idx = cnt * 4;
      for (int i = 0; i < remain; ++i) {
        float sum = 0.0;
        for (int k = cs; k < ce; ++k) {
          sum +=
              din_ptr[k * channel_size + idx] * din_ptr[k * channel_size + idx];
        }
        sum = k + sum * alpha;
        dout_ch_ptr[0] = din_ch_ptr[0] * pow(sum, -beta);
        dout_ch_ptr++;
        din_ch_ptr++;
        idx++;
      }
    }
  }
}

template <>
void compute_within_channels<float>(const float* din,
                                    float* dout,
                                    int num,
                                    int channel,
                                    int h,
                                    int w,
                                    int local_size,
                                    float alpha,
                                    float beta,
                                    float k) {
  LOG(ERROR) << "unsupported method!!";
  return;
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
