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

#include "lite/backends/arm/math/fp16/activation_fp16.h"
#include <algorithm>
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

template <>
void act_relu<float16_t>(const float16_t* din,
                         float16_t* dout,
                         int size,
                         int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 5;
  int neon_loop_rem = nums_per_thread & 31;
  int neon_loop_rem_cnt = neon_loop_rem >> 3;
  int neon_loop_rem_rem = neon_loop_rem & 7;
  int stride = neon_loop_rem_cnt << 3;
  float16x8_t vzero = vdupq_n_f16(0.f);

  LITE_PARALLEL_BEGIN(i, tid, threads) {
    const float16_t* ptr_in_thread = din + i * nums_per_thread;
    float16_t* ptr_out_thread = dout + i * nums_per_thread;
    for (int j = 0; j < neon_loop_cnt; j++) {
      float16x8_t vin0 = vld1q_f16(ptr_in_thread);
      float16x8_t vin1 = vld1q_f16(ptr_in_thread + 8);
      float16x8_t vin2 = vld1q_f16(ptr_in_thread + 16);
      float16x8_t vin3 = vld1q_f16(ptr_in_thread + 24);
      ptr_in_thread += 32;
      vst1q_f16(ptr_out_thread, vmaxq_f16(vin0, vzero));
      vst1q_f16(ptr_out_thread + 8, vmaxq_f16(vin1, vzero));
      vst1q_f16(ptr_out_thread + 16, vmaxq_f16(vin2, vzero));
      vst1q_f16(ptr_out_thread + 24, vmaxq_f16(vin3, vzero));
      ptr_out_thread += 32;
    }
    for (int j = 0; j < neon_loop_rem_cnt; j++) {
      float16x8_t vin0 = vld1q_f16(ptr_in_thread);
      ptr_in_thread += 8;
      vst1q_f16(ptr_out_thread, vmaxq_f16(vin0, vzero));
      ptr_out_thread += 8;
    }
    for (int j = 0; j < neon_loop_rem_rem; ++j) {
      ptr_out_thread[0] = ptr_in_thread[0] > 0.f ? ptr_in_thread[0] : 0.f;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END()
  float16_t* out_ptr_remain = dout + threads * nums_per_thread;
  const float16_t* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    out_ptr_remain[0] = in_ptr_remain[0] > 0.f ? in_ptr_remain[0] : 0.f;
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

template <>
void act_hard_sigmoid<float16_t>(const float16_t* din,
                                 float16_t* dout,
                                 const int size,
                                 const float slope,
                                 const float offset,
                                 int threads) {
  int cnt = size >> 5;
  int remain = size & 31;

  int cnt_4 = remain >> 3;
  int remain_4 = remain & 7;

  float16x8_t vzero_8 = vdupq_n_f16(float16_t(0));
  float16x8_t vone_8 = vdupq_n_f16(float16_t(1));
  float16x8_t vslope_8 = vdupq_n_f16(float16_t(slope));
  float16x8_t voffset_8 = vdupq_n_f16(float16_t(offset));
  for (int j = 0; j < cnt; j++) {
    float16x8_t vin0 = vld1q_f16(din);
    float16x8_t vin1 = vld1q_f16(din + 8);
    float16x8_t vin2 = vld1q_f16(din + 16);
    float16x8_t vin3 = vld1q_f16(din + 24);
    din += 32;
    float16x8_t vsum0 = vaddq_f16(voffset_8, vmulq_f16(vin0, vslope_8));
    float16x8_t vsum1 = vaddq_f16(voffset_8, vmulq_f16(vin1, vslope_8));
    float16x8_t vsum2 = vaddq_f16(voffset_8, vmulq_f16(vin2, vslope_8));
    float16x8_t vsum3 = vaddq_f16(voffset_8, vmulq_f16(vin3, vslope_8));
    float16x8_t vres0 = vbslq_f16(vcgtq_f16(vsum0, vzero_8), vsum0, vzero_8);
    float16x8_t vres1 = vbslq_f16(vcgtq_f16(vsum1, vzero_8), vsum1, vzero_8);
    float16x8_t vres2 = vbslq_f16(vcgtq_f16(vsum2, vzero_8), vsum2, vzero_8);
    float16x8_t vres3 = vbslq_f16(vcgtq_f16(vsum3, vzero_8), vsum3, vzero_8);

    vres0 = vbslq_f16(vcltq_f16(vres0, vone_8), vres0, vone_8);
    vres1 = vbslq_f16(vcltq_f16(vres1, vone_8), vres1, vone_8);
    vres2 = vbslq_f16(vcltq_f16(vres2, vone_8), vres2, vone_8);
    vres3 = vbslq_f16(vcltq_f16(vres3, vone_8), vres3, vone_8);
    vst1q_f16(dout, vres0);
    vst1q_f16(dout + 8, vres1);
    vst1q_f16(dout + 16, vres2);
    vst1q_f16(dout + 24, vres3);
    dout += 32;
  }
  for (int j = 0; j < cnt_4; j++) {
    float16x8_t vin0 = vld1q_f16(din);
    din += 8;
    float16x8_t vsum0 = vaddq_f16(voffset_8, vmulq_f16(vin0, vslope_8));
    float16x8_t vres0 = vbslq_f16(vcgtq_f16(vsum0, vzero_8), vsum0, vzero_8);
    vres0 = vbslq_f16(vcltq_f16(vres0, vone_8), vres0, vone_8);
    vst1q_f16(dout, vres0);
    dout += 8;
  }
  for (int64_t i = 0; i < remain_4; i++) {
    dout[0] = din[0] * slope + offset;
    dout[0] = dout[0] < 1.0f ? dout[0] : 1.0f;
    dout[0] = dout[0] > 0.0f ? dout[0] : 0.0f;
    ++din;
    ++dout;
  }
}

template <>
void act_hard_swish<float16_t>(const float16_t* din,
                               float16_t* dout,
                               const int size,
                               const float threshold,
                               const float scale,
                               const float offset,
                               int threads) {
  int cnt = size >> 5;
  int remain = size & 31;
  float scale_r = 1. / scale;

  int cnt_8 = remain >> 3;
  int remain_8 = remain & 7;

  float16x8_t vzero_8 = vdupq_n_f16(float16_t(0));
  float16x8_t vthreshold_8 = vdupq_n_f16(float16_t(threshold));
  float16x8_t vscale_8 = vdupq_n_f16(float16_t(scale_r));
  float16x8_t voffset_8 = vdupq_n_f16(float16_t(offset));

  for (int i = 0; i < cnt; i++) {
    float16x8_t vdin0 = vld1q_f16(din);
    float16x8_t vdin1 = vld1q_f16(din + 8);
    float16x8_t vdin2 = vld1q_f16(din + 16);
    float16x8_t vdin3 = vld1q_f16(din + 24);
    float16x8_t vtmp0 = vminq_f16(
        vthreshold_8, vmaxq_f16(vzero_8, vaddq_f16(vdin0, voffset_8)));
    float16x8_t vsum0 = vmulq_f16(vscale_8, vdin0);
    float16x8_t vtmp1 = vminq_f16(
        vthreshold_8, vmaxq_f16(vzero_8, vaddq_f16(vdin1, voffset_8)));
    float16x8_t vsum1 = vmulq_f16(vscale_8, vdin1);
    float16x8_t vtmp2 = vminq_f16(
        vthreshold_8, vmaxq_f16(vzero_8, vaddq_f16(vdin2, voffset_8)));
    float16x8_t vsum2 = vmulq_f16(vscale_8, vdin2);
    float16x8_t vtmp3 = vminq_f16(
        vthreshold_8, vmaxq_f16(vzero_8, vaddq_f16(vdin3, voffset_8)));
    float16x8_t vsum3 = vmulq_f16(vscale_8, vdin3);
    float16x8_t vres0 = vmulq_f16(vsum0, vtmp0);
    float16x8_t vres1 = vmulq_f16(vsum1, vtmp1);
    float16x8_t vres2 = vmulq_f16(vsum2, vtmp2);
    float16x8_t vres3 = vmulq_f16(vsum3, vtmp3);
    vst1q_f16(dout, vres0);
    vst1q_f16(dout + 8, vres1);
    vst1q_f16(dout + 16, vres2);
    vst1q_f16(dout + 24, vres3);
    din += 32;
    dout += 32;
  }
  for (int i = 0; i < cnt_8; i++) {
    float16x8_t vdin0 = vld1q_f16(din);
    din += 8;
    float16x8_t vtmp0 = vminq_f16(
        vthreshold_8, vmaxq_f16(vzero_8, vaddq_f16(vdin0, voffset_8)));
    float16x8_t vsum0 = vmulq_f16(vscale_8, vdin0);
    float16x8_t vres0 = vmulq_f16(vsum0, vtmp0);
    vst1q_f16(dout, vres0);
    dout += 8;
  }
  for (int i = 0; i < remain_8; i++) {
    dout[0] =
        std::min(std::max(0.f, din[0] + offset), threshold) * din[0] * scale_r;
    din++;
    dout++;
  }
}

template <>
void act_prelu<float16_t>(const float16_t* din,
                          float16_t* dout,
                          int outer_size,
                          int channel_size,
                          int inner_size,
                          std::string mode,
                          const float16_t* alpha_data,
                          int threads) {
  int stride_size = inner_size * channel_size;
  int cnt = inner_size >> 5;
  int remain = inner_size & 31;
  int cnt_8 = remain >> 3;
  int rem_8 = remain & 7;
  float16x8_t vzero = vdupq_n_f16(0.f);
  if (mode == "all" || mode == "channel") {
    for (int n = 0; n < outer_size; n++) {
      const float16_t* data_in_batch = din + n * stride_size;
      float16_t* data_out_batch = dout + n * stride_size;

      LITE_PARALLEL_BEGIN(c, tid, channel_size) {
        const float16_t* data_in_c = data_in_batch + c * inner_size;
        float16_t* data_out_c = data_out_batch + c * inner_size;

        float16_t slope = (mode == "all") ? alpha_data[0] : alpha_data[c];
        float16x8_t vslope = vdupq_n_f16(slope);
        for (int i = 0; i < cnt; i++) {
          float16x8_t vin0 = vld1q_f16(data_in_c);
          float16x8_t vin1 = vld1q_f16(data_in_c + 8);
          float16x8_t vin2 = vld1q_f16(data_in_c + 16);
          float16x8_t vin3 = vld1q_f16(data_in_c + 24);
          data_in_c += 32;
          float16x8_t vres0 =
              vbslq_f16(vcgtq_f16(vin0, vzero), vin0, vmulq_f16(vin0, vslope));
          float16x8_t vres1 =
              vbslq_f16(vcgtq_f16(vin1, vzero), vin1, vmulq_f16(vin1, vslope));
          float16x8_t vres2 =
              vbslq_f16(vcgtq_f16(vin2, vzero), vin2, vmulq_f16(vin2, vslope));
          float16x8_t vres3 =
              vbslq_f16(vcgtq_f16(vin3, vzero), vin3, vmulq_f16(vin3, vslope));
          vst1q_f16(data_out_c, vres0);
          vst1q_f16(data_out_c + 8, vres1);
          vst1q_f16(data_out_c + 16, vres2);
          vst1q_f16(data_out_c + 24, vres3);
          data_out_c += 32;
        }
        for (int i = 0; i < cnt_8; i++) {
          float16x8_t vin0 = vld1q_f16(data_in_c);
          data_in_c += 8;
          float16x8_t vres0 =
              vbslq_f16(vcgtq_f16(vin0, vzero), vin0, vmulq_f16(vin0, vslope));
          vst1q_f16(data_out_c, vres0);
          data_out_c += 8;
        }
        for (int i = rem_8; i > 0; i--) {
          *(data_out_c++) =
              data_in_c[0] > 0.f ? data_in_c[0] : data_in_c[0] * slope;
          data_in_c++;
        }
      }
      LITE_PARALLEL_END()
    }
  } else {  // mode = element
    for (int n = 0; n < outer_size; n++) {
      const float16_t* data_in_batch = din + n * stride_size;
      const float16_t* data_alpha_batch = alpha_data + n * stride_size;
      float16_t* data_out_batch = dout + n * stride_size;

      LITE_PARALLEL_BEGIN(c, tid, channel_size) {
        const float16_t* data_in_c = data_in_batch + c * inner_size;
        const float16_t* data_alpha_c = data_alpha_batch + c * inner_size;
        float16_t* data_out_c = data_out_batch + c * inner_size;
        for (int i = 0; i < cnt; i++) {
          float16x8_t vin0 = vld1q_f16(data_in_c);
          float16x8_t valpha0 = vld1q_f16(data_alpha_c);
          float16x8_t vin1 = vld1q_f16(data_in_c + 8);
          float16x8_t valpha1 = vld1q_f16(data_alpha_c + 8);
          float16x8_t vin2 = vld1q_f16(data_in_c + 16);
          float16x8_t valpha2 = vld1q_f16(data_alpha_c + 16);
          float16x8_t vin3 = vld1q_f16(data_in_c + 24);
          float16x8_t valpha3 = vld1q_f16(data_alpha_c + 24);
          data_in_c += 32;
          data_alpha_c += 32;
          float16x8_t vres0 =
              vbslq_f16(vcgtq_f16(vin0, vzero), vin0, vmulq_f16(vin0, valpha0));
          float16x8_t vres1 =
              vbslq_f16(vcgtq_f16(vin1, vzero), vin1, vmulq_f16(vin1, valpha1));
          float16x8_t vres2 =
              vbslq_f16(vcgtq_f16(vin2, vzero), vin2, vmulq_f16(vin2, valpha2));
          float16x8_t vres3 =
              vbslq_f16(vcgtq_f16(vin3, vzero), vin3, vmulq_f16(vin3, valpha3));
          vst1q_f16(data_out_c, vres0);
          vst1q_f16(data_out_c + 8, vres1);
          vst1q_f16(data_out_c + 16, vres2);
          vst1q_f16(data_out_c + 24, vres3);
          data_out_c += 32;
        }
        for (int i = 0; i < cnt_8; i++) {
          float16x8_t vin0 = vld1q_f16(data_in_c);
          float16x8_t valpha0 = vld1q_f16(data_alpha_c);
          data_in_c += 8;
          data_alpha_c += 8;
          float16x8_t vres0 =
              vbslq_f16(vcgtq_f16(vin0, vzero), vin0, vmulq_f16(vin0, valpha0));
          vst1q_f16(data_out_c, vres0);
          data_out_c += 8;
        }
        for (int i = 0; i < rem_8; i++) {
          data_out_c[0] = data_in_c[0] > 0.f ? data_in_c[0]
                                             : data_in_c[0] * data_alpha_c[0];
          data_in_c++;
          data_alpha_c++;
          data_out_c++;
        }
      }
      LITE_PARALLEL_END()
    }
  }
}

// tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <>
void act_tanh<float16_t>(const float16_t* din,
                         float16_t* dout,
                         int size,
                         int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim8 = nums_per_thread >> 3;
  int neon_loop_remain_dim8 = nums_per_thread - (neon_loop_cnt_dim8 << 3);
  float16x8_t vmax_f16 = vdupq_n_f16(70.00008f);
  float16x8_t vmin_f16 = vdupq_n_f16(-70.00008f);
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    const float16_t* ptr_in_thread = din + i * nums_per_thread;
    float16_t* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim8; ++k) {
      float16x8_t data = vld1q_f16(ptr_in_thread);
      data = vminq_f16(data, vmax_f16);
      data = vmaxq_f16(data, vmin_f16);
      float16x8_t exp_plus_vec = expq_ps_f16(data);
      float16x8_t exp_minus_vec = expq_ps_f16(vnegq_f16(data));
      float16x8_t exp_sum_vec = vaddq_f16(exp_plus_vec, exp_minus_vec);
      float16x8_t exp_diff_vec = vsubq_f16(exp_plus_vec, exp_minus_vec);
      float16x8_t recip = divq_ps_f16(exp_diff_vec, exp_sum_vec);
      vst1q_f16(ptr_out_thread, recip);
      ptr_out_thread += 8;
      ptr_in_thread += 8;
    }
    for (int j = 0; j < neon_loop_remain_dim8; ++j) {
      ptr_out_thread[0] = (expf(ptr_in_thread[0]) - expf(-ptr_in_thread[0])) /
                          (expf(ptr_in_thread[0]) + expf(-ptr_in_thread[0]));
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float16_t* ptr_out = dout + threads * nums_per_thread;
  const float16_t* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = (expf(ptr_in[0]) - expf(-ptr_in[0])) /
                 (expf(ptr_in[0]) + expf(-ptr_in[0]));
    ptr_in++;
    ptr_out++;
  }
}

template <>
void act_sigmoid<float16_t>(const float16_t* din,
                            float16_t* dout,
                            int size,
                            int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim8 = nums_per_thread >> 3;
  int neon_loop_remain_dim8 = nums_per_thread - (neon_loop_cnt_dim8 << 3);

  float16x8_t vzero = vdupq_n_f16(0.f);
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    float16x8_t exp_vec = vdupq_n_f16(0.0f);
    float16x8_t recip = vdupq_n_f16(0.0f);
    const float16_t* ptr_in_thread = din + i * nums_per_thread;
    float16_t* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim8; ++k) {
      exp_vec = expq_ps_f16(vnegq_f16(vld1q_f16(ptr_in_thread)));
      exp_vec = vaddq_f16(exp_vec, vdupq_n_f16(1.0f));
      recip = vrecpeq_f16(exp_vec);
      recip = vmulq_f16(vrecpsq_f16(exp_vec, recip), recip);
      recip = vmulq_f16(vrecpsq_f16(exp_vec, recip), recip);
      vst1q_f16(ptr_out_thread, recip);
      ptr_out_thread += 8;
      ptr_in_thread += 8;
    }
    for (int j = 0; j < neon_loop_remain_dim8; ++j) {
      ptr_out_thread[0] = 1.f / (1 + expf(-ptr_in_thread[0]));
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float16_t* ptr_out = dout + threads * nums_per_thread;
  const float16_t* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = 1.f / (1 + expf(-ptr_in[0]));
    ptr_in++;
    ptr_out++;
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
