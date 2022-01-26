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

#include "lite/backends/arm/math/activation.h"
#include <algorithm>
#include <string>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void act_relu<float>(const float* din, float* dout, int size, int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 4;
  int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
  float32x4_t vzero = vdupq_n_f32(0.f);
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    int cnt = neon_loop_cnt;
#ifdef __aarch64__
    for (int num = 0; num < neon_loop_cnt; ++num) {
      float32x4_t vr0 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr1 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr2 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr3 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      vr0 = vmaxq_f32(vr0, vzero);
      vr1 = vmaxq_f32(vr1, vzero);
      vr2 = vmaxq_f32(vr2, vzero);
      vr3 = vmaxq_f32(vr3, vzero);
      vst1q_f32(ptr_out_thread, vr0);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vr1);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vr2);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vr3);
      ptr_out_thread += 4;
    }

#else
    if (cnt > 0) {
      asm volatile(
          "1:                                     @ loop header\n"
          "vld1.32  {d0-d3}, [%[din]]!            @ load din 0\n"
          "vld1.32  {d4-d7}, [%[din]]!            @ load din 0\n"

          "vmax.f32 q8, q0, %q[vzero]             @ relu\n"
          "vmax.f32 q9, q1, %q[vzero]             @ relu\n"
          "vmax.f32 q10, q2, %q[vzero]            @ relu\n"
          "vmax.f32 q11, q3, %q[vzero]            @ relu\n"

          "vst1.32  {d16-d19}, [%[dout]]!         @ store result, add pointer\n"
          "vst1.32  {d20-d23}, [%[dout]]!         @ store result, add pointer\n"

          "subs %[cnt], #1                        @ loop count minus 1\n"
          "bne    1b                              @ jump to main loop start "
          "point\n"
          : [dout] "+r"(ptr_out_thread),
            [din] "+r"(ptr_in_thread),
            [cnt] "+r"(cnt)
          : [vzero] "w"(vzero)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
    }
#endif
    for (int j = 0; j < neon_loop_remain; ++j) {
      ptr_out_thread[0] = ptr_in_thread[0] > 0.f ? ptr_in_thread[0] : 0.f;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float* out_ptr_remain = dout + threads * nums_per_thread;
  const float* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    out_ptr_remain[0] = in_ptr_remain[0] > 0.f ? in_ptr_remain[0] : 0.f;
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

template <>
void act_relu_neg<float>(const float* din,
                         float* dout,
                         int size,
                         float negative_slope,
                         int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 4;
  int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
  float32x4_t vzero = vdupq_n_f32(0.f);
  float32x4_t valpha = vdupq_n_f32(negative_slope);
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    int cnt = neon_loop_cnt;
#ifdef __aarch64__
    for (int num = 0; num < neon_loop_cnt; ++num) {
      float32x4_t vr0 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr1 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr2 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr3 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;

      uint32x4_t vm0 = vcgeq_f32(vr0, vzero);
      uint32x4_t vm1 = vcgeq_f32(vr1, vzero);
      uint32x4_t vm2 = vcgeq_f32(vr2, vzero);
      uint32x4_t vm3 = vcgeq_f32(vr3, vzero);

      float32x4_t vn0 = vmulq_f32(vr0, valpha);
      float32x4_t vn1 = vmulq_f32(vr1, valpha);
      float32x4_t vn2 = vmulq_f32(vr2, valpha);
      float32x4_t vn3 = vmulq_f32(vr3, valpha);

      float32x4_t vo0 = vbslq_f32(vm0, vr0, vn0);
      float32x4_t vo1 = vbslq_f32(vm1, vr1, vn1);
      float32x4_t vo2 = vbslq_f32(vm2, vr2, vn2);
      float32x4_t vo3 = vbslq_f32(vm3, vr3, vn3);

      vst1q_f32(ptr_out_thread, vo0);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vo1);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vo2);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vo3);
      ptr_out_thread += 4;
    }

#else
    if (cnt > 0) {
      asm volatile(
          "1:                                             @ loop header\n"
          "vld1.32  {d0-d3}, [%[din]]!            @ load din 0\n"
          "vld1.32  {d4-d7}, [%[din]]!            @ load din 0\n"

          "vcge.f32 q8, q0, %q[vzero]             @ get mask\n"
          "vcge.f32 q9, q1, %q[vzero]             @ get mask\n"
          "vcge.f32 q10, q2, %q[vzero]            @ get mask\n"
          "vcge.f32 q11, q3, %q[vzero]            @ get mask\n"

          "vmul.f32   q4, q0, %q[valpha]          @ get neg data\n"
          "vmul.f32   q5, q1, %q[valpha]          @ get neg data\n"
          "vmul.f32   q6, q2, %q[valpha]          @ get neg data\n"
          "vmul.f32   q7, q3, %q[valpha]          @ get neg data\n"

          "vbit   q4, q0, q8                      @ bitsel, insert q0 to q4, "
          "if q8 is 1\n"
          "vbit   q5, q1, q9                      @ bitsel, insert q1 to q5, "
          "if q9 is 1\n"
          "vbit   q6, q2, q10                     @ bitsel, insert q2 to q6, "
          "if q10 is 1\n"
          "vbit   q7, q3, q11                     @ bitsel, insert q3 to q7, "
          "if q11 is 1\n"

          "vst1.32  {d8-d11}, [%[dout]]!          @ store result, add pointer\n"
          "vst1.32  {d12-d15}, [%[dout]]!         @ store result, add pointer\n"

          "subs %[cnt], #1                        @ loop count minus 1\n"
          "bne    1b                              @ jump to main loop start "
          "point\n"
          : [dout] "+r"(ptr_out_thread),
            [din] "+r"(ptr_in_thread),
            [cnt] "+r"(cnt)
          : [vzero] "w"(vzero), [valpha] "w"(valpha)
          : "cc",
            "memory",
            "q0",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11");
    }
#endif
    for (int j = 0; j < neon_loop_remain; ++j) {
      ptr_out_thread[0] = ptr_in_thread[0] > 0.f
                              ? ptr_in_thread[0]
                              : ptr_in_thread[0] * negative_slope;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float* out_ptr_remain = dout + threads * nums_per_thread;
  const float* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    out_ptr_remain[0] = in_ptr_remain[0] > 0.f
                            ? in_ptr_remain[0]
                            : in_ptr_remain[0] * negative_slope;
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

template <>
void act_clipped_relu<float>(
    const float* din, float* dout, int size, float coef, int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 4;
  int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
  float32x4_t vzero = vdupq_n_f32(0.f);
  float32x4_t vclip = vdupq_n_f32(coef);
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    int cnt = neon_loop_cnt;
#ifdef __aarch64__
    for (int num = 0; num < neon_loop_cnt; ++num) {
      float32x4_t vr0 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr1 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr2 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vr3 = vld1q_f32(ptr_in_thread);
      ptr_in_thread += 4;
      float32x4_t vt0 = vmaxq_f32(vr0, vzero);
      float32x4_t vt1 = vmaxq_f32(vr1, vzero);
      float32x4_t vt2 = vmaxq_f32(vr2, vzero);
      float32x4_t vt3 = vmaxq_f32(vr3, vzero);

      float32x4_t vo0 = vminq_f32(vt0, vclip);
      float32x4_t vo1 = vminq_f32(vt1, vclip);
      float32x4_t vo2 = vminq_f32(vt2, vclip);
      float32x4_t vo3 = vminq_f32(vt3, vclip);

      vst1q_f32(ptr_out_thread, vo0);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vo1);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vo2);
      ptr_out_thread += 4;
      vst1q_f32(ptr_out_thread, vo3);
      ptr_out_thread += 4;
    }
#else
    if (cnt > 0) {
      asm volatile(
          "1:                                     @ loop header\n"
          "vld1.32  {d0-d3}, [%[din]]!            @ load din 0\n"
          "vld1.32  {d4-d7}, [%[din]]!            @ load din 0\n"

          "vmax.f32 q8, q0, %q[vzero]             @ relu\n"
          "vmax.f32 q9, q1, %q[vzero]             @ relu\n"
          "vmax.f32 q10, q2, %q[vzero]            @ relu\n"
          "vmax.f32 q11, q3, %q[vzero]            @ relu\n"

          "vmin.f32 q4, q8, %q[vclip]             @ clip relu\n"
          "vmin.f32 q5, q9, %q[vclip]             @ clip relu\n"
          "vmin.f32 q6, q10, %q[vclip]            @ clip relu\n"
          "vmin.f32 q7, q11, %q[vclip]            @ clip relu\n"

          "vst1.32  {d8-d11}, [%[dout]]!          @ store result, add pointer\n"
          "vst1.32  {d12-d15}, [%[dout]]!         @ store result, add pointer\n"

          "subs %[cnt], #1                        @ loop count minus 1\n"
          "bne    1b                              @ jump to main loop start "
          "point\n"
          : [dout] "+r"(ptr_out_thread),
            [din] "+r"(ptr_in_thread),
            [cnt] "+r"(cnt)
          : [vzero] "w"(vzero), [vclip] "w"(vclip)
          : "cc",
            "memory",
            "q0",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "q8",
            "q9",
            "q10",
            "q11");
    }
#endif
    for (int j = 0; j < neon_loop_remain; ++j) {
      ptr_out_thread[0] = ptr_in_thread[0] > 0.f ? ptr_in_thread[0] : 0.f;
      ptr_out_thread[0] = ptr_out_thread[0] < coef ? ptr_out_thread[0] : coef;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float* out_ptr_remain = dout + threads * nums_per_thread;
  const float* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    out_ptr_remain[0] = in_ptr_remain[0] > 0.f ? in_ptr_remain[0] : 0.f;
    out_ptr_remain[0] = out_ptr_remain[0] < coef ? out_ptr_remain[0] : coef;
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

template <>
void act_prelu<float>(const float* din,
                      float* dout,
                      int outer_size,
                      int channel_size,
                      int inner_size,
                      std::string mode,
                      const float* alpha_data,
                      int threads) {
  if (mode == "all" || mode == "channel") {
    int stride_size = inner_size * channel_size;
    int cnt = inner_size >> 4;
    int remain = inner_size & 15;
    float32x4_t vzero = vdupq_n_f32(0.f);
    for (int n = 0; n < outer_size; n++) {
      const float* data_in_batch = din + n * stride_size;
      float* data_out_batch = dout + n * stride_size;
      LITE_PARALLEL_BEGIN(c, tid, channel_size) {
        const float* data_in_c = data_in_batch + c * inner_size;
        float* data_out_c = data_out_batch + c * inner_size;

        float slope = mode == "all" ? alpha_data[0] : alpha_data[c];
        float32x4_t vslope = vdupq_n_f32(slope);
#ifdef __aarch64__
        for (int i = 0; i < cnt; ++i) {
          float32x4_t vr0 = vld1q_f32(data_in_c);
          float32x4_t vr1 = vld1q_f32(data_in_c + 4);
          float32x4_t vr2 = vld1q_f32(data_in_c + 8);
          float32x4_t vr3 = vld1q_f32(data_in_c + 12);
          uint32x4_t vm0 = vcltq_f32(vr0, vzero);    // vr0 <= vzero
          uint32x4_t vm1 = vcltq_f32(vr1, vzero);    // vr0 <= vzero
          uint32x4_t vm2 = vcltq_f32(vr2, vzero);    // vr0 <= vzero
          uint32x4_t vm3 = vcltq_f32(vr3, vzero);    // vr0 <= vzero
          float32x4_t vo0 = vmulq_f32(vr0, vslope);  // vr0 * vslope
          float32x4_t vo1 = vmulq_f32(vr1, vslope);  // vr0 * vslope
          float32x4_t vo2 = vmulq_f32(vr2, vslope);  // vr0 * vslope
          float32x4_t vo3 = vmulq_f32(vr3, vslope);  // vr0 * vslope
          float32x4_t vos0 = vbslq_f32(vm0, vo0, vr0);
          float32x4_t vos1 = vbslq_f32(vm1, vo1, vr1);
          float32x4_t vos2 = vbslq_f32(vm2, vo2, vr2);
          float32x4_t vos3 = vbslq_f32(vm3, vo3, vr3);
          vst1q_f32(data_out_c, vos0);
          vst1q_f32(data_out_c + 4, vos1);
          vst1q_f32(data_out_c + 8, vos2);
          vst1q_f32(data_out_c + 12, vos3);
          data_in_c += 16;
          data_out_c += 16;
        }
#else
        int cnt_loop = cnt;
        if (cnt_loop > 0) {
          asm volatile(
              "vld1.32    {d0-d3}, [%[ptr_in]]!                       @ load "
              "input to q0, q1\n"
              "pld [%[ptr_in]]                                @ preload\n"
              "pld [%[ptr_in], #64]                           @ preload\n"
              "pld [%[ptr_in], #128]                          @ preload\n"
              "pld [%[ptr_in], #192]                          @ preload\n"
              "1:                                             @main loop\n"
              "vld1.32    {d4-d7}, [%[ptr_in]]!               @ load input to "
              "q2, q3\n"
              "vclt.f32   q8, q0, %q[vzero]                   @vcle q0 <= "
              "vzero\n"
              "vclt.f32   q9, q1, %q[vzero]                   @vcle q1 <= "
              "vzero\n"
              "vmul.f32  q10, q0, %q[vslope]                  @vmul q0 * "
              "vslope\n"
              "vmul.f32  q11, q1, %q[vslope]                  @vmul q1 * "
              "vslope\n"

              "vclt.f32  q12, q2, %q[vzero]                   @vcle q2 <= "
              "vzero\n"
              "vclt.f32  q13, q3, %q[vzero]                   @vcle q3 <= "
              "vzero\n"
              "vmul.f32  q14, q2, %q[vslope]                  @vmul q2 * "
              "vslope\n"
              "vmul.f32  q15, q3, %q[vslope]                  @vmul q3 * "
              "vslope\n"

              "vbif.32    q10, q0, q8                         @vbit q10, q0, "
              "q8\n"
              "vbif.32    q11, q1, q9                         @vbit q11, q1, "
              "q9\n"
              "vbif.32    q14, q2, q12                        @vbit q14, q2, "
              "q12\n"
              "vbif.32    q15, q3, q13                        @vbit q15, q3, "
              "q13\n"

              "subs       %[cnt], #1                          @subs nn, 1\n"
              "vld1.32    {d0-d3}, [%[ptr_in]]!               @ load input to "
              "q0, q1\n"

              "vst1.f32   {d20-d23}, [%[dout]]!               @store data\n"
              "vst1.f32   {d28-d31}, [%[dout]]!               @store data\n"
              "bne        1b                                  @bne nn\n"
              "sub    %[ptr_in], #32                          @ ptr-32\n"
              : [ptr_in] "+r"(data_in_c),
                [cnt] "+r"(cnt_loop),
                [dout] "+r"(data_out_c)
              : [vzero] "w"(vzero), [vslope] "w"(vslope)
              : "cc",
                "memory",
                "q0",
                "q1",
                "q2",
                "q3",
                "q8",
                "q9",
                "q10",
                "q11",
                "q12",
                "q13",
                "q14",
                "q15");
        }
#endif  // __aarch64__
        for (int i = remain; i > 0; i--) {
          *(data_out_c++) =
              data_in_c[0] > 0.f ? data_in_c[0] : data_in_c[0] * slope;
          data_in_c++;
        }
      }
      LITE_PARALLEL_END();
    }
  } else {  // mode = element
    int stride_size = inner_size * channel_size;
    for (int n = 0; n < outer_size; n++) {
      const float* data_in_batch = din + n * stride_size;
      const float* data_alpha_batch = alpha_data + n * stride_size;
      float* data_out_batch = dout + n * stride_size;
      for (int c = 0; c < channel_size; c++) {
        const float* data_in_c = data_in_batch + c * inner_size;
        const float* data_alpha_c = data_alpha_batch + c * inner_size;
        float* data_out_c = data_out_batch + c * inner_size;
        for (int i = 0; i < inner_size; i++) {
          data_out_c[0] = data_in_c[0] > 0.f ? data_in_c[0]
                                             : data_in_c[0] * data_alpha_c[0];
          data_in_c++;
          data_alpha_c++;
          data_out_c++;
        }
      }
    }
  }
}

template <>
void act_sigmoid<float>(const float* din, float* dout, int size, int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);

  float32x4_t vzero = vdupq_n_f32(0.f);
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    float32x4_t exp_vec = vdupq_n_f32(0.0f);
    float32x4_t recip = vdupq_n_f32(0.0f);
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim4; ++k) {
      exp_vec = exp_ps(vnegq_f32(vld1q_f32(ptr_in_thread)));
      exp_vec = vaddq_f32(exp_vec, vdupq_n_f32(1.0f));
      recip = vrecpeq_f32(exp_vec);
      recip = vmulq_f32(vrecpsq_f32(exp_vec, recip), recip);
      recip = vmulq_f32(vrecpsq_f32(exp_vec, recip), recip);
      vst1q_f32(ptr_out_thread, recip);
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain_dim4; ++j) {
      ptr_out_thread[0] = 1.f / (1 + expf(-ptr_in_thread[0]));
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = 1.f / (1 + expf(-ptr_in[0]));
    ptr_in++;
    ptr_out++;
  }
}

// tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <>
void act_tanh<float>(const float* din, float* dout, int size, int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    float32x4_t exp_plus_vec = vdupq_n_f32(0.0f);
    float32x4_t exp_minus_vec = vdupq_n_f32(0.0f);
    float32x4_t exp_sum_vec = vdupq_n_f32(0.0f);
    float32x4_t exp_diff_vec = vdupq_n_f32(0.0f);
    float32x4_t recip = vdupq_n_f32(0.0f);
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim4; ++k) {
      float32x4_t data = vld1q_f32(ptr_in_thread);
      data = vminq_f32(data, vdupq_n_f32(70.00008f));
      data = vmaxq_f32(data, vdupq_n_f32(-70.00008f));
      exp_plus_vec = exp_ps(data);
      exp_minus_vec = exp_ps(vnegq_f32(data));
      exp_sum_vec = vaddq_f32(exp_plus_vec, exp_minus_vec);
      exp_diff_vec = vsubq_f32(exp_plus_vec, exp_minus_vec);
      recip = div_ps(exp_diff_vec, exp_sum_vec);
      vst1q_f32(ptr_out_thread, recip);
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain_dim4; ++j) {
      ptr_out_thread[0] = (expf(ptr_in_thread[0]) - expf(-ptr_in_thread[0])) /
                          (expf(ptr_in_thread[0]) + expf(-ptr_in_thread[0]));
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = (expf(ptr_in[0]) - expf(-ptr_in[0])) /
                 (expf(ptr_in[0]) + expf(-ptr_in[0]));
    ptr_in++;
    ptr_out++;
  }
}

// swish: x /(1 + exp(-(b * x)))
template <>
void act_swish<float>(
    const float* din, float* dout, int size, float coef, int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
  const float beta = coef;
  float32x4_t vbeta = vdupq_n_f32(beta);
  float32x4_t vone = vdupq_n_f32(1.f);
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim4; ++k) {
      float32x4_t va = vld1q_f32(ptr_in_thread);             // x
      float32x4_t vb = vnegq_f32(vld1q_f32(ptr_in_thread));  // -x
      float32x4_t vsum = vmulq_f32(vb, vbeta);
      vsum = exp_ps(vsum);
      float32x4_t vc = vaddq_f32(vone, vsum);
      float32x4_t vrst = div_ps(va, vc);
      vst1q_f32(ptr_out_thread, vrst);
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain_dim4; ++j) {
      ptr_out_thread[0] =
          ptr_in_thread[0] / (1.0 + expf(-ptr_in_thread[0] * beta));
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = ptr_in[0] / (1.0 + expf(-ptr_in[0] * beta));
    ptr_in++;
    ptr_out++;
  }
}

template <>
void act_log<float>(const float* din, float* dout, int size, int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);

  float32x4_t vzero = vdupq_n_f32(0.f);
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    float32x4_t exp_vec = vdupq_n_f32(0.0f);
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim4; ++k) {
      exp_vec = log_ps(vld1q_f32(ptr_in_thread));
      vst1q_f32(ptr_out_thread, exp_vec);
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain_dim4; ++j) {
      ptr_out_thread[0] = logf(ptr_in_thread[0]);
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = logf(ptr_in[0]);
    ptr_in++;
    ptr_out++;
  }
}

template <>
void act_exp<float>(const float* din, float* dout, int size, int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);

  float32x4_t vzero = vdupq_n_f32(0.f);
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    float32x4_t exp_vec = vdupq_n_f32(0.0f);
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim4; ++k) {
      exp_vec = exp_ps(vld1q_f32(ptr_in_thread));
      vst1q_f32(ptr_out_thread, exp_vec);
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < neon_loop_remain_dim4; ++j) {
      ptr_out_thread[0] = expf(ptr_in_thread[0]);
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    ptr_out[0] = expf(ptr_in[0]);
    ptr_in++;
    ptr_out++;
  }
}

template <>
void act_floor<float>(const float* din, float* dout, int size, int threads) {
  const float* ptr_in = din;
  float* ptr_out = dout;
  for (int i = 0; i < size; ++i) {
    ptr_out[0] = floorf(ptr_in[0]);
    ptr_in++;
    ptr_out++;
  }
}

template <>
void act_hard_sigmoid<float>(const float* din,
                             float* dout,
                             const int64_t size,
                             const float slope,
                             const float offset,
                             int threads) {
  int cnt_4 = size >> 4;
  int remain_4 = size & 15;

  float32x4_t vzero_4 = vdupq_n_f32(0.f);
  float32x4_t vone_4 = vdupq_n_f32(1.f);

  float32x4_t vslope_4 = vdupq_n_f32(slope);
  float32x4_t voffset_4 = vdupq_n_f32(offset);
  for (int64_t i = 0; i < cnt_4; ++i) {
    float32x4_t vr0 = vld1q_f32(din);
    float32x4_t vr1 = vld1q_f32(din + 4);
    float32x4_t vr2 = vld1q_f32(din + 8);
    float32x4_t vr3 = vld1q_f32(din + 12);

    float32x4_t vres0 = vmulq_f32(vr0, vslope_4);  // vr0 * vslope
    float32x4_t vres1 = vmulq_f32(vr1, vslope_4);  // vr1 * vslope
    float32x4_t vres2 = vmulq_f32(vr2, vslope_4);  // vr2 * vslope
    float32x4_t vres3 = vmulq_f32(vr3, vslope_4);  // vr3 * vslope

    vres0 = vaddq_f32(vres0, voffset_4);  // vres0 += offset
    vres1 = vaddq_f32(vres1, voffset_4);  // vres1 += offset
    vres2 = vaddq_f32(vres2, voffset_4);  // vres2 += offset
    vres3 = vaddq_f32(vres3, voffset_4);  // vres3 += offset

    uint32x4_t vm_gt0_0 = vcgtq_f32(vres0, vzero_4);  // vres0 > zero
    uint32x4_t vm_gt0_1 = vcgtq_f32(vres1, vzero_4);  // vres1 > zero
    uint32x4_t vm_gt0_2 = vcgtq_f32(vres2, vzero_4);  // vres2 > zero
    uint32x4_t vm_gt0_3 = vcgtq_f32(vres3, vzero_4);  // vres3 > zero

    uint32x4_t vm_lt1_0 = vcltq_f32(vres0, vone_4);  // vres0 < 1
    uint32x4_t vm_lt1_1 = vcltq_f32(vres1, vone_4);  // vres1 < 1
    uint32x4_t vm_lt1_2 = vcltq_f32(vres2, vone_4);  // vres2 < 1
    uint32x4_t vm_lt1_3 = vcltq_f32(vres3, vone_4);  // vres3 < 1

    float32x4_t vos0 =
        vbslq_f32(vm_gt0_0, vres0, vzero_4);  // vos0 = vres0 > 0? vres0: 0
    float32x4_t vos1 = vbslq_f32(vm_gt0_1, vres1, vzero_4);
    float32x4_t vos2 = vbslq_f32(vm_gt0_2, vres2, vzero_4);
    float32x4_t vos3 = vbslq_f32(vm_gt0_3, vres3, vzero_4);

    vos0 = vbslq_f32(vm_lt1_0, vos0, vone_4);  // vos0 = vos0 < 1? vres0: 1
    vos1 = vbslq_f32(vm_lt1_1, vos1, vone_4);
    vos2 = vbslq_f32(vm_lt1_2, vos2, vone_4);
    vos3 = vbslq_f32(vm_lt1_3, vos3, vone_4);

    vst1q_f32(dout, vos0);
    vst1q_f32(dout + 4, vos1);
    vst1q_f32(dout + 8, vos2);
    vst1q_f32(dout + 12, vos3);

    dout += 16;
    din += 16;
  }
  for (int64_t i = 0; i < remain_4; ++i) {
    dout[0] = din[0] * slope + offset;
    dout[0] = dout[0] < 1.0f ? dout[0] : 1.0f;
    dout[0] = dout[0] > 0.0f ? dout[0] : 0.0f;
    ++din;
    ++dout;
  }
}

template <>
void act_rsqrt<float>(const float* din, float* dout, int size, int threads) {
  const float* ptr_in = din;
  float* ptr_out = dout;
  for (int i = 0; i < size; ++i) {
    ptr_out[0] = 1.0 / sqrtf(ptr_in[0]);
    ptr_in++;
    ptr_out++;
  }
}

template <>
void act_sqrt<float>(const float* din, float* dout, int size, int threads) {
  const float* ptr_in = din;
  float* ptr_out = dout;
  for (int i = 0; i < size; ++i) {
    ptr_out[0] = sqrtf(ptr_in[0]);
    ptr_in++;
    ptr_out++;
  }
}

template <>
void act_square<float>(const float* din, float* dout, int size, int threads) {
  const float* ptr_in = din;
  float* ptr_out = dout;
  for (int i = 0; i < size; ++i) {
    ptr_out[0] = ptr_in[0] * ptr_in[0];
    ptr_in++;
    ptr_out++;
  }
}

template <>
void act_hard_swish<float>(const float* din,
                           float* dout,
                           int size,
                           float threshold,
                           float scale,
                           float offset,
                           int threads) {
  int nums_per_thread = size / threads;
  int remain = size - nums_per_thread * threads;
  int neon_loop_cnt_dim4 = nums_per_thread >> 2;
  int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);

  const float* ptr_in = din;
  float* ptr_out = dout;
  float scale_r = 1. / scale;
  float32x4_t scale_v, offset_v, threshold_v, zero;
  offset_v = vdupq_n_f32(offset);
  scale_v = vdupq_n_f32(scale_r);
  zero = vdupq_n_f32(0.);
  threshold_v = vdupq_n_f32(threshold);

  LITE_PARALLEL_BEGIN(i, tid, threads) {
    const float* ptr_in_thread = ptr_in + i * nums_per_thread;
    float* ptr_out_thread = ptr_out + i * nums_per_thread;
    for (int j = 0; j < neon_loop_cnt_dim4; j++) {
      float32x4_t in = vld1q_f32(ptr_in_thread);
      float32x4_t in_add_offset = vaddq_f32(in, offset_v);
      float32x4_t tmp1 = vmaxq_f32(zero, in_add_offset);
      float32x4_t tmp2 = vminq_f32(threshold_v, tmp1);
      float32x4_t tmp3 = vmulq_f32(scale_v, in);
      float32x4_t tmp4 = vmulq_f32(tmp2, tmp3);
      vst1q_f32(ptr_out_thread, tmp4);
      ptr_in_thread += 4;
      ptr_out_thread += 4;
    }

    for (int j = 0; j < neon_loop_remain_dim4; j++) {
      ptr_out_thread[0] =
          std::min(std::max(0.f, ptr_in_thread[0] + offset), threshold) *
          ptr_in_thread[0] * scale_r;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  ptr_out = dout + threads * nums_per_thread;
  ptr_in = din + threads * nums_per_thread;
  for (int i = 0; i < remain; i++) {
    ptr_out[0] = std::min(std::max(0.f, ptr_in[0] + offset), threshold) *
                 ptr_in[0] * scale_r;
    ptr_in++;
    ptr_out++;
  }
}

template <>
void act_reciprocal<float>(const float* din,
                           float* dout,
                           int size,
                           int threads) {
  const float* ptr_in = din;
  float* ptr_out = dout;
  for (int i = 0; i < size; ++i) {
    ptr_out[0] = 1.0 / ptr_in[0];
    ptr_in++;
    ptr_out++;
  }
}

template <>
void act_abs<float>(const float* din, float* dout, int size, int threads) {
  for (int i = 0; i < size; ++i) {
    dout[0] = (din[0] > 0 ? din[0] : -din[0]);
    din++;
    dout++;
  }
}

template <typename T>
void erf(const T* din, T* dout, int size, int threads) {
  for (int i = 0; i < size; ++i) {
    dout[0] = std::erf(din[0]);
    din++;
    dout++;
  }
}

template void erf<float>(const float* din, float* dout, int size, int threads);

template <typename T>
void sign(const T* din, T* dout, int size, int threads) {
  for (int i = 0; i < size; ++i) {
    dout[0] = (dout[0] >= (T)0) - ((T)0 >= dout[0]);
    din++;
    dout++;
  }
}

template void sign<float>(const float* din, float* dout, int size, int threads);

template <typename T>
void softplus(const T* din, T* dout, int size, float beta, int threads) {
  for (int i = 0; i < size; ++i) {
    dout[0] = 1. / beta * log((T)1. + exp(din[0] * beta));
    din++;
    dout++;
  }
}

template void softplus<float>(
    const float* din, float* dout, int size, float beta, int threads);

template <>
void act_thresholded_relu<float>(
    const float* din, float* dout, int size, float threshold, int threads) {
  for (int i = 0; i < size; ++i) {
    dout[0] = (din[0] > threshold ? din[0] : 0.f);
    din++;
    dout++;
  }
}

// elu: out = max(0,x) + min(0, alpha *(exp(x) - 1)
template <>
void act_elu<float>(
    const float* din, float* dout, int size, float alpha, int threads) {
  int nums_per_thread = size / threads;
  int thread_remain = size % threads;
  int neon_loop_cnt_dim16 = nums_per_thread >> 4;
  int neon_loop_remain_dim16 = nums_per_thread & 15;
  float32x4_t valpha = vdupq_n_f32(alpha);
  float32x4_t vzero = vdupq_n_f32(0.f);
  float32x4_t vone = vdupq_n_f32(1.f);
  int cnt = neon_loop_remain_dim16 >> 2;
  int remain = neon_loop_remain_dim16 & 3;
  LITE_PARALLEL_BEGIN(i, tid, threads) {
    const float* ptr_in_thread = din + i * nums_per_thread;
    float* ptr_out_thread = dout + i * nums_per_thread;
    for (int k = 0; k < neon_loop_cnt_dim16; ++k) {
      float32x4_t va = vld1q_f32(ptr_in_thread);
      float32x4_t vb = vld1q_f32(ptr_in_thread + 4);
      float32x4_t vc = vld1q_f32(ptr_in_thread + 8);
      float32x4_t vd = vld1q_f32(ptr_in_thread + 12);

      uint32x4_t gt_a = vcgtq_f32(va, vzero);
      uint32x4_t gt_b = vcgtq_f32(vb, vzero);
      uint32x4_t gt_c = vcgtq_f32(vc, vzero);
      uint32x4_t gt_d = vcgtq_f32(vd, vzero);

      float32x4_t va_exp = exp_ps(va);
      float32x4_t vb_exp = exp_ps(vb);
      float32x4_t vc_exp = exp_ps(vc);
      float32x4_t vd_exp = exp_ps(vd);
      float32x4_t va_sub = vsubq_f32(va_exp, vone);
      float32x4_t vb_sub = vsubq_f32(vb_exp, vone);
      float32x4_t vc_sub = vsubq_f32(vc_exp, vone);
      float32x4_t vd_sub = vsubq_f32(vd_exp, vone);
      va_sub = vmulq_f32(va_sub, valpha);
      vb_sub = vmulq_f32(vb_sub, valpha);
      vc_sub = vmulq_f32(vc_sub, valpha);
      vd_sub = vmulq_f32(vd_sub, valpha);

      float32x4_t va_rst = vbslq_f32(gt_a, va, va_sub);
      float32x4_t vb_rst = vbslq_f32(gt_b, vb, vb_sub);
      float32x4_t vc_rst = vbslq_f32(gt_c, vc, vc_sub);
      float32x4_t vd_rst = vbslq_f32(gt_d, vd, vd_sub);

      vst1q_f32(ptr_out_thread, va_rst);
      vst1q_f32(ptr_out_thread + 4, vb_rst);
      vst1q_f32(ptr_out_thread + 8, vc_rst);
      vst1q_f32(ptr_out_thread + 12, vd_rst);
      ptr_out_thread += 16;
      ptr_in_thread += 16;
    }
    for (int j = 0; j < cnt; j++) {
      float32x4_t va = vld1q_f32(ptr_in_thread);
      uint32x4_t gt_0 = vcgtq_f32(va, vzero);

      float32x4_t va_exp = exp_ps(va);
      float32x4_t va_sub = vsubq_f32(va_exp, vone);
      va_sub = vmulq_f32(va_sub, valpha);

      float32x4_t va_rst = vbslq_f32(gt_0, va, va_sub);
      vst1q_f32(ptr_out_thread, va_rst);
      ptr_out_thread += 4;
      ptr_in_thread += 4;
    }
    for (int j = 0; j < remain; j++) {
      ptr_out_thread[0] = ptr_in_thread[0] > 0.f
                              ? ptr_in_thread[0]
                              : (alpha * (expf(ptr_in_thread[0]) - 1));
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  LITE_PARALLEL_END();
  float* ptr_out = dout + threads * nums_per_thread;
  const float* ptr_in = din + threads * nums_per_thread;
  for (int j = 0; j < thread_remain; j++) {
    ptr_out[0] = ptr_in[0] > 0.f ? ptr_in[0] : (alpha * (expf(ptr_in[0]) - 1));
    ptr_in++;
    ptr_out++;
  }
}

// when using approximation
// $out = \\frac{1}{2}x(1+tanh(\\sqrt{\\frac{2}{\\pi}}(x+0.044715x^{3}))$
// or else
// $out = \\frac{1 + erf(\\frac{x}{\\sqrt{2}})}{2} x$
template <>
void act_gelu<float>(
    const float* din, float* dout, int size, bool approximate, int threads) {
  if (approximate) {
    const float pi = std::atan(1) * 4;
    const float sqrt_2_div_pi = std::sqrt(2 / pi);
    for (int i = 0; i < size; i++) {
      float x = *din;
      *dout = 0.5 * x *
              (1 + std::tanh(sqrt_2_div_pi * (x + 0.044715 * std::pow(x, 3))));
      ++din;
      ++dout;
    }
  } else {
    const float sqrt_2 = std::sqrt(2.0);
    for (int i = 0; i < size; i++) {
      float x = *din;
      *dout = 0.5 * x * (1 + std::erf(x / sqrt_2));
      ++din;
      ++dout;
    }
  }
}
template <>
void mish(const float* din, float* dout, int size, float threshold) {
  int cnt = size >> 4;
  int remain = size & 15;
  float32x4_t vthreshold = vdupq_n_f32(threshold);
  float32x4_t vone = vdupq_n_f32(1.f);
  float32x4_t minus_vthreshold = vdupq_n_f32(-threshold);
  for (int64_t i = 0; i < cnt; i++) {
    float32x4_t vx0 = vld1q_f32(din);
    float32x4_t vx4 = vld1q_f32(din + 4);
    float32x4_t vx8 = vld1q_f32(din + 8);
    float32x4_t vx12 = vld1q_f32(din + 12);

    uint32x4_t gt_0 = vcgtq_f32(vx0, vthreshold);
    uint32x4_t gt_4 = vcgtq_f32(vx4, vthreshold);
    uint32x4_t gt_8 = vcgtq_f32(vx8, vthreshold);
    uint32x4_t gt_12 = vcgtq_f32(vx12, vthreshold);

    uint32x4_t lt_0 = vcltq_f32(vx0, minus_vthreshold);
    uint32x4_t lt_4 = vcltq_f32(vx4, minus_vthreshold);
    uint32x4_t lt_8 = vcltq_f32(vx8, minus_vthreshold);
    uint32x4_t lt_12 = vcltq_f32(vx12, minus_vthreshold);

    float32x4_t data0 = vminq_f32(vx0, vdupq_n_f32(70.00008f));  // e^x
    float32x4_t data4 = vminq_f32(vx4, vdupq_n_f32(70.00008f));
    float32x4_t data8 = vminq_f32(vx8, vdupq_n_f32(70.00008f));
    float32x4_t data12 = vminq_f32(vx12, vdupq_n_f32(70.00008f));
    data0 = vmaxq_f32(data0, vdupq_n_f32(-70.00008f));
    data4 = vmaxq_f32(data4, vdupq_n_f32(-70.00008f));
    data8 = vmaxq_f32(data8, vdupq_n_f32(-70.00008f));
    data12 = vmaxq_f32(data12, vdupq_n_f32(-70.00008f));

    float32x4_t vleftx0 = exp_ps(data0);
    float32x4_t vleftx4 = exp_ps(data4);
    float32x4_t vleftx8 = exp_ps(data8);
    float32x4_t vleftx12 = exp_ps(data12);

    float32x4_t vmiddlex0 = log_ps(vaddq_f32(vleftx0, vone));  // ln(1+e^x)
    float32x4_t vmiddlex4 = log_ps(vaddq_f32(vleftx4, vone));
    float32x4_t vmiddlex8 = log_ps(vaddq_f32(vleftx8, vone));
    float32x4_t vmiddlex12 = log_ps(vaddq_f32(vleftx12, vone));

    float32x4_t sp0 = vbslq_f32(gt_0, vx0, vmiddlex0);
    float32x4_t sp4 = vbslq_f32(gt_4, vx4, vmiddlex4);
    float32x4_t sp8 = vbslq_f32(gt_8, vx8, vmiddlex8);
    float32x4_t sp12 = vbslq_f32(gt_12, vx12, vmiddlex12);

    sp0 = vbslq_f32(lt_0, vleftx0, sp0);
    sp4 = vbslq_f32(lt_4, vleftx4, sp4);
    sp8 = vbslq_f32(lt_8, vleftx8, sp8);
    sp12 = vbslq_f32(lt_12, vleftx12, sp12);

    sp0 = vminq_f32(sp0, vdupq_n_f32(70.00008f));
    sp4 = vminq_f32(sp4, vdupq_n_f32(70.00008f));
    sp8 = vminq_f32(sp8, vdupq_n_f32(70.00008f));
    sp12 = vminq_f32(sp12, vdupq_n_f32(70.00008f));
    sp0 = vmaxq_f32(sp0, vdupq_n_f32(-70.00008f));
    sp4 = vmaxq_f32(sp4, vdupq_n_f32(-70.00008f));
    sp8 = vmaxq_f32(sp8, vdupq_n_f32(-70.00008f));
    sp12 = vmaxq_f32(sp12, vdupq_n_f32(-70.00008f));

    float32x4_t exp_sp0 = exp_ps(sp0);
    float32x4_t exp_sp4 = exp_ps(sp4);
    float32x4_t exp_sp8 = exp_ps(sp8);
    float32x4_t exp_sp12 = exp_ps(sp12);

    float32x4_t exp_minussp0 = exp_ps(vnegq_f32(sp0));
    float32x4_t exp_minussp4 = exp_ps(vnegq_f32(sp4));
    float32x4_t exp_minussp8 = exp_ps(vnegq_f32(sp8));
    float32x4_t exp_minussp12 = exp_ps(vnegq_f32(sp12));

    float32x4_t exp_sum0 = vaddq_f32(exp_sp0, exp_minussp0);
    float32x4_t exp_sum4 = vaddq_f32(exp_sp4, exp_minussp4);
    float32x4_t exp_sum8 = vaddq_f32(exp_sp8, exp_minussp8);
    float32x4_t exp_sum12 = vaddq_f32(exp_sp12, exp_minussp12);

    float32x4_t exp_diff0 = vsubq_f32(exp_sp0, exp_minussp0);
    float32x4_t exp_diff4 = vsubq_f32(exp_sp4, exp_minussp4);
    float32x4_t exp_diff8 = vsubq_f32(exp_sp8, exp_minussp8);
    float32x4_t exp_diff12 = vsubq_f32(exp_sp12, exp_minussp12);

    float32x4_t res0 = vmulq_f32(vx0, div_ps(exp_diff0, exp_sum0));
    float32x4_t res4 = vmulq_f32(vx4, div_ps(exp_diff4, exp_sum4));
    float32x4_t res8 = vmulq_f32(vx8, div_ps(exp_diff8, exp_sum8));
    float32x4_t res12 = vmulq_f32(vx12, div_ps(exp_diff12, exp_sum12));

    vst1q_f32(dout, res0);
    vst1q_f32(dout + 4, res4);
    vst1q_f32(dout + 8, res8);
    vst1q_f32(dout + 12, res12);
    dout += 16;
    din += 16;
  }
  for (int i = 0; i < remain; i++) {
    float x = din[i];
    float sp = 0.0f;
    if (threshold > 0 && x > threshold)
      sp = x;
    else if (threshold > 0 && x < -threshold)
      sp = expf(x);
    else
      sp = log1pf(expf(x));
    dout[i] = x * std::tanh(sp);
  }
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
