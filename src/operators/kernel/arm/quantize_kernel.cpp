/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef QUANT_OP

#include "operators/kernel/quantize_kernel.h"
#include <cmath>

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>

#ifndef __aarch64__
inline float32_t vmaxvq_f32(float32x4_t r) {
  float32x2_t v = vmax_f32(vget_high_f32(r), vget_low_f32(r));
  return vget_lane_f32(vpmax_f32(v, v), 0);
}
#endif

inline int32x4_t vrnd_towards_zero(float32x4_t r) { return vcvtq_s32_f32(r); }

inline int32x4_t vrnd_away_zero(float32x4_t r) {
  float32x4_t plus = vdupq_n_f32(0.5);
  float32x4_t minus = vdupq_n_f32(-0.5);
  float32x4_t zero = vdupq_n_f32(0);
  uint32x4_t more_than_zero = vcgtq_f32(r, zero);
  float32x4_t temp = vbslq_f32(more_than_zero, plus, minus);
  temp = vaddq_f32(r, temp);
  int32x4_t ret = vcvtq_s32_f32(temp);
  return ret;
}

inline int32x4_t vrnd_to_even(float32x4_t r) {
#if 0
  int32x4_t ret;
  float value[4];
  vst1q_f32(value, r);
  for (int i = 0; i < 4; ++i) {
    float v = round(value[i]);
    int32_t q = (int32_t)v;
    if (abs(abs(v - value[i]) - 0.5) > 0) {
      ret[i] = q;
    } else {
      if (abs(q) % 2 == 0) {
        ret[i] = q;
      } else {
        ret[i] = q + ((q > 0) ? -1 : 1);
      }
    }
  }
  return ret;
#else
  float32x4_t point5 = vdupq_n_f32(0.5);
  int32x4_t one = vdupq_n_s32(1);
  int32x4_t zero = vdupq_n_s32(0);

  int32x4_t rnd = vrnd_away_zero(r);
  float32x4_t frnd = vcvtq_f32_s32(rnd);
  frnd = vsubq_f32(frnd, r);
  frnd = vabsq_f32(frnd);
  uint32x4_t equal_point5 = vceqq_f32(frnd, point5);
  int32x4_t abs_rnd = vabsq_s32(rnd);
  abs_rnd = vandq_s32(abs_rnd, one);
  uint32x4_t not_mod2 = vreinterpretq_u32_s32(abs_rnd);
  uint32x4_t mask = vandq_u32(equal_point5, not_mod2);
  uint32x4_t more_than_zero = vcgtq_s32(rnd, zero);
  more_than_zero = vandq_u32(more_than_zero, vreinterpretq_u32_s32(one));
  mask = veorq_u32(more_than_zero, mask);
  more_than_zero = veorq_u32(more_than_zero, vreinterpretq_u32_s32(one));
  mask = vaddq_u32(more_than_zero, mask);
  int32x4_t smask = vreinterpretq_s32_u32(mask);
  smask = vsubq_s32(smask, one);
  rnd = vaddq_s32(rnd, smask);
  return rnd;
#endif
}

namespace paddle_mobile {
namespace operators {

static float find_abs_max(const Tensor *input) {
  float max_abs = 0.f;
  const float *x = input->data<const float>();
  size_t size = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  size_t loop = size >> 4;
  size_t remain = size & 0xF;
  for (size_t i = 0; i < loop; ++i) {
    float32x4_t max;
    float32x4_t r0 = vld1q_f32(x);
    float32x4_t r1 = vld1q_f32(x + 4);
    float32x4_t r2 = vld1q_f32(x + 8);
    float32x4_t r3 = vld1q_f32(x + 12);
    r0 = vabsq_f32(r0);
    r1 = vabsq_f32(r1);
    r2 = vabsq_f32(r2);
    r3 = vabsq_f32(r3);
    max[0] = vmaxvq_f32(r0);
    max[1] = vmaxvq_f32(r1);
    max[2] = vmaxvq_f32(r2);
    max[3] = vmaxvq_f32(r3);
    max[0] = vmaxvq_f32(max);
    if (max[0] > max_abs) {
      max_abs = max[0];
    }
    x += 16;
  }
  size = remain;
#endif
  for (size_t i = 0; i < size; ++i) {
    float value = std::abs(x[i]);
    if (value > max_abs) {
      max_abs = value;
    }
  }
  return max_abs;
}

#ifdef __aarch64__
static void quantize_round_to_even(const Tensor *input, const float scale,
                                   Tensor *output) {
  const float *x = input->data<const float>();
  int8_t *y = output->mutable_data<int8_t>();
  size_t size = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  size_t loop = size >> 4;
  size_t remain = size & 0xF;

  #pragma omp parallel for
  for (size_t i = 0; i < loop; ++i) {
    const float *local_x = x + (i << 4);
    int8_t *local_y = y + (i << 4);
    float32x4_t r0 = vld1q_f32(local_x);
    float32x4_t r1 = vld1q_f32(local_x + 4);
    float32x4_t r2 = vld1q_f32(local_x + 8);
    float32x4_t r3 = vld1q_f32(local_x + 12);
    r0 = vmulq_n_f32(r0, scale);
    r1 = vmulq_n_f32(r1, scale);
    r2 = vmulq_n_f32(r2, scale);
    r3 = vmulq_n_f32(r3, scale);
    int32x4_t q0 = vrnd_to_even(r0);
    int32x4_t q1 = vrnd_to_even(r1);
    int32x4_t q2 = vrnd_to_even(r2);
    int32x4_t q3 = vrnd_to_even(r3);
    int16x4_t d0 = vmovn_s32(q0);
    int16x4_t d1 = vmovn_s32(q1);
    int16x4_t d2 = vmovn_s32(q2);
    int16x4_t d3 = vmovn_s32(q3);
    int16x8_t q5 = vcombine_s16(d0, d1);
    int16x8_t q6 = vcombine_s16(d2, d3);
    int8x8_t d5 = vmovn_s16(q5);
    int8x8_t d6 = vmovn_s16(q6);
    vst1_s8(local_y, d5);
    vst1_s8(local_y + 8, d6);
  }
  size = remain;
  x += (loop << 4);
  y += (loop << 4);
#endif
  for (size_t i = 0; i < size; ++i) {
    float value = x[i] * scale;
    float v = round(value);
    int32_t q = (int32_t)v;
    if (abs(abs(q - value) - 0.5) > 0) {
      y[i] = q;
    } else {
      if (abs(q) % 2 == 0) {
        y[i] = q;
      } else {
        y[i] = q + ((q > 0) ? -1 : 1);
      }
    }
  }
}

static void quantize_round_to_zero(const Tensor *input, const float scale,
                                   Tensor *output) {
  const float *x = input->data<const float>();
  int8_t *y = output->mutable_data<int8_t>();
  size_t size = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  size_t loop = size >> 4;
  size_t remain = size & 0xF;

  #pragma omp parallel for
  for (size_t i = 0; i < loop; ++i) {
    const float *local_x = x + (i << 4);
    int8_t *local_y = y + (i << 4);
    float32x4_t r0 = vld1q_f32(local_x);
    float32x4_t r1 = vld1q_f32(local_x + 4);
    float32x4_t r2 = vld1q_f32(local_x + 8);
    float32x4_t r3 = vld1q_f32(local_x + 12);
    r0 = vmulq_n_f32(r0, scale);
    r1 = vmulq_n_f32(r1, scale);
    r2 = vmulq_n_f32(r2, scale);
    r3 = vmulq_n_f32(r3, scale);
    int32x4_t q0 = vrnd_towards_zero(r0);
    int32x4_t q1 = vrnd_towards_zero(r1);
    int32x4_t q2 = vrnd_towards_zero(r2);
    int32x4_t q3 = vrnd_towards_zero(r3);
    int16x4_t d0 = vmovn_s32(q0);
    int16x4_t d1 = vmovn_s32(q1);
    int16x4_t d2 = vmovn_s32(q2);
    int16x4_t d3 = vmovn_s32(q3);
    int16x8_t q5 = vcombine_s16(d0, d1);
    int16x8_t q6 = vcombine_s16(d2, d3);
    int8x8_t d5 = vmovn_s16(q5);
    int8x8_t d6 = vmovn_s16(q6);
    vst1_s8(local_y, d5);
    vst1_s8(local_y + 8, d6);
  }
  size = remain;
  x += (loop << 4);
  y += (loop << 4);
#endif
  for (size_t i = 0; i < size; ++i) {
    y[i] = static_cast<int8_t>(x[i] * scale);
  }
}

static void quantize_round_to_nearest(const Tensor *input, const float scale,
                                      Tensor *output) {
  const float *x = input->data<const float>();
  int8_t *y = output->mutable_data<int8_t>();
  size_t size = input->numel();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  size_t loop = size >> 4;
  size_t remain = size & 0xF;

  #pragma omp parallel for
  for (size_t i = 0; i < loop; ++i) {
    const float *local_x = x + (i << 4);
    int8_t *local_y = y + (i << 4);
    float32x4_t r0 = vld1q_f32(local_x);
    float32x4_t r1 = vld1q_f32(local_x + 4);
    float32x4_t r2 = vld1q_f32(local_x + 8);
    float32x4_t r3 = vld1q_f32(local_x + 12);
    r0 = vmulq_n_f32(r0, scale);
    r1 = vmulq_n_f32(r1, scale);
    r2 = vmulq_n_f32(r2, scale);
    r3 = vmulq_n_f32(r3, scale);
    int32x4_t q0 = vrnd_away_zero(r0);
    int32x4_t q1 = vrnd_away_zero(r1);
    int32x4_t q2 = vrnd_away_zero(r2);
    int32x4_t q3 = vrnd_away_zero(r3);
    int16x4_t d0 = vmovn_s32(q0);
    int16x4_t d1 = vmovn_s32(q1);
    int16x4_t d2 = vmovn_s32(q2);
    int16x4_t d3 = vmovn_s32(q3);
    int16x8_t q5 = vcombine_s16(d0, d1);
    int16x8_t q6 = vcombine_s16(d2, d3);
    int8x8_t d5 = vmovn_s16(q5);
    int8x8_t d6 = vmovn_s16(q6);
    vst1_s8(local_y, d5);
    vst1_s8(local_y + 8, d6);
  }
  size = remain;
  x += (loop << 4);
  y += (loop << 4);
#endif
  for (size_t i = 0; i < size; ++i) {
    y[i] = round(x[i] * scale);
  }
}
#else   // __aarch64__

static void quantize_round_to_even(const Tensor *input, const float scale,
                                   const std::vector<int> &paddings,
                                   const int8_t padding_val, Tensor *output) {}

static void quantize_round_to_nearest(const Tensor *input, const float scale,
                                      const std::vector<int> &paddings,
                                      const int8_t padding_val,
                                      Tensor *output) {}

static void quantize_round_to_zero(const Tensor *input, const float scale,
                                   const std::vector<int> &paddings,
                                   const int8_t padding_val, Tensor *output) {
  int channels = input->dims()[1];
  int input_h = input->dims()[2];
  int input_w = input->dims()[3];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];
  int input_spatial_size = input_h * input_w;
  int output_spatial_size = output_h * output_w;
  const float *x = input->data<float>();
  int8_t *y = output->mutable_data<int8_t>();
  // valid area start
  int start = paddings[0] * output_w + paddings[1];

  for (int batch = 0; batch < input->dims()[0]; ++batch) {
    #pragma omp parallel for
    for (int c = 0; c < channels - 3; c += 4) {
      const float *input0 = x + (batch * channels + c) * input_spatial_size;
      const float *input1 = input0 + input_spatial_size;
      const float *input2 = input1 + input_spatial_size;
      const float *input3 = input2 + input_spatial_size;
      size_t offset = (batch * channels + c) * output_spatial_size;
      for (int h = 0; h < 2; ++h) {
        int8_t *y0 =
            y + offset + h * ((input_h + paddings[0]) * output_w - paddings[1]);
        int8_t *y1 = y0 + output_spatial_size;
        int8_t *y2 = y1 + output_spatial_size;
        int8_t *y3 = y2 + output_spatial_size;
        int loop = start >> 4;
        int remain = start & 0xF;
        asm volatile(
            "vdup.s8    q0,     %[val]      \n"
            "cmp        %[loop], #0         \n"
            "ble        start_remain_%=     \n"

            "store_16w_%=:                  \n"
            "vst1.32    {q0}, [%[y0]]!      \n"
            "vst1.32    {q0}, [%[y1]]!      \n"
            "vst1.32    {q0}, [%[y2]]!      \n"
            "vst1.32    {q0}, [%[y3]]!      \n"
            "subs       %[loop], #1         \n"
            "bne        store_16w_%=        \n"

            "start_remain_%=:               \n"
            "cmp        %[remain], #8       \n"
            "blt        store_4w_%=         \n"
            "vst1.32    {d0}, [%[y0]]!      \n"
            "vst1.32    {d0}, [%[y1]]!      \n"
            "vst1.32    {d0}, [%[y2]]!      \n"
            "vst1.32    {d0}, [%[y3]]!      \n"
            "sub        %[remain], #8       \n"

            "store_4w_%=:                   \n"
            "cmp        %[remain], #4       \n"
            "blt        store_2w_%=         \n"
            "vst1.32    {d0[0]}, [%[y0]]!   \n"
            "vst1.32    {d0[0]}, [%[y1]]!   \n"
            "vst1.32    {d0[0]}, [%[y2]]!   \n"
            "vst1.32    {d0[0]}, [%[y3]]!   \n"
            "sub        %[remain], #4       \n"

            "store_2w_%=:                   \n"
            "cmp        %[remain], #4       \n"
            "blt        store_1w_%=         \n"
            "vst1.16    {d0[0]}, [%[y0]]!   \n"
            "vst1.16    {d0[0]}, [%[y1]]!   \n"
            "vst1.16    {d0[0]}, [%[y2]]!   \n"
            "vst1.16    {d0[0]}, [%[y3]]!   \n"
            "sub        %[remain], #2       \n"

            "store_1w_%=:                   \n"
            "cmp        %[remain], #1       \n"
            "blt        end_%=              \n"
            "vst1.8     {d0[0]}, [%[y0]]!   \n"
            "vst1.8     {d0[0]}, [%[y1]]!   \n"
            "vst1.8     {d0[0]}, [%[y2]]!   \n"
            "vst1.8     {d0[0]}, [%[y3]]!   \n"
            "end_%=:                        \n"
            : [y0] "+r"(y0), [y1] "+r"(y1), [y2] "+r"(y2), [y3] "+r"(y3),
              [loop] "+r"(loop), [remain] "+r"(remain)
            : [val] "r"(padding_val)
            : "cc", "memory", "q0");
      }
      // quantize valid area
      int8_t *y0 = y + offset + start;
      int8_t *y1 = y0 + output_spatial_size;
      int8_t *y2 = y1 + output_spatial_size;
      int8_t *y3 = y2 + output_spatial_size;
      for (int h = 0; h < input_h; ++h) {
        const float *x0 = input0 + h * input_w;
        const float *x1 = input1 + h * input_w;
        const float *x2 = input2 + h * input_w;
        const float *x3 = input3 + h * input_w;
        int loop = input_w >> 4;
        int remain = input_w & 0xF;
        int pad_loop = paddings[1] >> 1;  // (paddings[1] << 1) >> 2
        int pad_remain = (paddings[1] << 1) & 0x3;
        int remain_steps = remain;
        asm volatile(
            "vdup.f32   q0, %[scale]        \n"
            "cmp        %[loop], #0         \n"
            "ble        quantize_remain_%=  \n"

            "loop_quantize_%=:              \n"
            "vld1.32    {q1, q2}, [%[x0]]!  \n"
            "vld1.32    {q3, q4}, [%[x1]]!  \n"
            "vld1.32    {q5, q6}, [%[x2]]!  \n"
            "vld1.32    {q7, q8}, [%[x3]]!  \n"
            "vmul.f32  q1, q1, q0           \n"
            "vmul.f32  q2, q2, q0           \n"
            "vmul.f32  q3, q3, q0           \n"
            "vmul.f32  q4, q4, q0           \n"
            "vmul.f32  q5, q5, q0           \n"
            "vmul.f32  q6, q6, q0           \n"
            "vmul.f32  q7, q7, q0           \n"
            "vmul.f32  q8, q8, q0           \n"
            "vcvt.s32.f32  q1, q1           \n"
            "vcvt.s32.f32  q2, q2           \n"
            "vcvt.s32.f32  q3, q3           \n"
            "vcvt.s32.f32  q4, q4           \n"
            "vcvt.s32.f32  q5, q5           \n"
            "vcvt.s32.f32  q6, q6           \n"
            "vcvt.s32.f32  q7, q7           \n"
            "vcvt.s32.f32  q8, q8           \n"
            "vmovn.s32  d2, q1              \n"
            "vmovn.s32  d3, q2              \n"
            "vmovn.s32  d4, q3              \n"
            "vmovn.s32  d5, q4              \n"
            "vmovn.s32  d6, q5              \n"
            "vmovn.s32  d7, q6              \n"
            "vmovn.s32  d8, q7              \n"
            "vmovn.s32  d9, q8              \n"
            "vmovn.s16  d18, q1             \n"
            "vmovn.s16  d20, q2             \n"
            "vmovn.s16  d22, q3             \n"
            "vmovn.s16  d24, q4             \n"
            "vld1.32    {q1, q2}, [%[x0]]!  \n"
            "vld1.32    {q3, q4}, [%[x1]]!  \n"
            "vld1.32    {q5, q6}, [%[x2]]!  \n"
            "vld1.32    {q7, q8}, [%[x3]]!  \n"
            "vmul.f32  q1, q1, q0           \n"
            "vmul.f32  q2, q2, q0           \n"
            "vmul.f32  q3, q3, q0           \n"
            "vmul.f32  q4, q4, q0           \n"
            "vmul.f32  q5, q5, q0           \n"
            "vmul.f32  q6, q6, q0           \n"
            "vmul.f32  q7, q7, q0           \n"
            "vmul.f32  q8, q8, q0           \n"
            "vcvt.s32.f32  q1, q1           \n"
            "vcvt.s32.f32  q2, q2           \n"
            "vcvt.s32.f32  q3, q3           \n"
            "vcvt.s32.f32  q4, q4           \n"
            "vcvt.s32.f32  q5, q5           \n"
            "vcvt.s32.f32  q6, q6           \n"
            "vcvt.s32.f32  q7, q7           \n"
            "vcvt.s32.f32  q8, q8           \n"
            "vmovn.s32  d2, q1              \n"
            "vmovn.s32  d3, q2              \n"
            "vmovn.s32  d4, q3              \n"
            "vmovn.s32  d5, q4              \n"
            "vmovn.s32  d6, q5              \n"
            "vmovn.s32  d7, q6              \n"
            "vmovn.s32  d8, q7              \n"
            "vmovn.s32  d9, q8              \n"
            "vmovn.s16  d19, q1             \n"
            "vmovn.s16  d21, q2             \n"
            "vmovn.s16  d23, q3             \n"
            "vmovn.s16  d25, q4             \n"
            "vst1.32    {q9}, [%[y0]]!      \n"
            "vst1.32    {q10}, [%[y1]]!     \n"
            "vst1.32    {q11}, [%[y2]]!     \n"
            "vst1.32    {q12}, [%[y3]]!     \n"

            "subs       %[loop], #1         \n"
            "bne        loop_quantize_%=    \n"

            "quantize_remain_%=:            \n"
            "cmp        %[remain], #0       \n"
            "ble        end_%=              \n"

            "vld1.32    {q1, q2}, [%[x0]]!  \n"
            "vld1.32    {q3, q4}, [%[x1]]!  \n"
            "vld1.32    {q5, q6}, [%[x2]]!  \n"
            "vld1.32    {q7, q8}, [%[x3]]!  \n"
            "vmul.f32  q1, q1, q0           \n"
            "vmul.f32  q2, q2, q0           \n"
            "vmul.f32  q3, q3, q0           \n"
            "vmul.f32  q4, q4, q0           \n"
            "vmul.f32  q5, q5, q0           \n"
            "vmul.f32  q6, q6, q0           \n"
            "vmul.f32  q7, q7, q0           \n"
            "vmul.f32  q8, q8, q0           \n"
            "vcvt.s32.f32  q1, q1           \n"
            "vcvt.s32.f32  q2, q2           \n"
            "vcvt.s32.f32  q3, q3           \n"
            "vcvt.s32.f32  q4, q4           \n"
            "vcvt.s32.f32  q5, q5           \n"
            "vcvt.s32.f32  q6, q6           \n"
            "vcvt.s32.f32  q7, q7           \n"
            "vcvt.s32.f32  q8, q8           \n"
            "vmovn.s32  d2, q1              \n"
            "vmovn.s32  d3, q2              \n"
            "vmovn.s32  d4, q3              \n"
            "vmovn.s32  d5, q4              \n"
            "vmovn.s32  d6, q5              \n"
            "vmovn.s32  d7, q6              \n"
            "vmovn.s32  d8, q7              \n"
            "vmovn.s32  d9, q8              \n"
            "vmovn.s16  d18, q1             \n"
            "vmovn.s16  d20, q2             \n"
            "vmovn.s16  d22, q3             \n"
            "vmovn.s16  d24, q4             \n"
            "vld1.32    {q1, q2}, [%[x0]]   \n"
            "vld1.32    {q3, q4}, [%[x1]]   \n"
            "vld1.32    {q5, q6}, [%[x2]]   \n"
            "vld1.32    {q7, q8}, [%[x3]]   \n"
            "vmul.f32  q1, q1, q0           \n"
            "vmul.f32  q2, q2, q0           \n"
            "vmul.f32  q3, q3, q0           \n"
            "vmul.f32  q4, q4, q0           \n"
            "vmul.f32  q5, q5, q0           \n"
            "vmul.f32  q6, q6, q0           \n"
            "vmul.f32  q7, q7, q0           \n"
            "vmul.f32  q8, q8, q0           \n"
            "vcvt.s32.f32  q1, q1           \n"
            "vcvt.s32.f32  q2, q2           \n"
            "vcvt.s32.f32  q3, q3           \n"
            "vcvt.s32.f32  q4, q4           \n"
            "vcvt.s32.f32  q5, q5           \n"
            "vcvt.s32.f32  q6, q6           \n"
            "vcvt.s32.f32  q7, q7           \n"
            "vcvt.s32.f32  q8, q8           \n"
            "vmovn.s32  d2, q1              \n"
            "vmovn.s32  d3, q2              \n"
            "vmovn.s32  d4, q3              \n"
            "vmovn.s32  d5, q4              \n"
            "vmovn.s32  d6, q5              \n"
            "vmovn.s32  d7, q6              \n"
            "vmovn.s32  d8, q7              \n"
            "vmovn.s32  d9, q8              \n"
            "vmovn.s16  d19, q1             \n"
            "vmovn.s16  d21, q2             \n"
            "vmovn.s16  d23, q3             \n"
            "vmovn.s16  d25, q4             \n"

            "cmp        %[remain], #8       \n"
            "blt        store_4w_%=         \n"
            "vst1.32    {d18}, [%[y0]]!     \n"
            "vst1.32    {d20}, [%[y1]]!     \n"
            "vst1.32    {d22}, [%[y2]]!     \n"
            "vst1.32    {d24}, [%[y3]]!     \n"
            "vmov.32    d18, d19            \n"
            "vmov.32    d20, d21            \n"
            "vmov.32    d22, d23            \n"
            "vmov.32    d24, d25            \n"
            "sub        %[remain], #8       \n"

            "store_4w_%=:                   \n"
            "cmp        %[remain], #4       \n"
            "blt        store_2w_%=         \n"
            "vst1.32    {d18[0]}, [%[y0]]!  \n"
            "vst1.32    {d20[0]}, [%[y1]]!  \n"
            "vst1.32    {d22[0]}, [%[y2]]!  \n"
            "vst1.32    {d24[0]}, [%[y3]]!  \n"
            "vext.32    d18, d18, d18, #1   \n"
            "vext.32    d20, d20, d20, #1   \n"
            "vext.32    d22, d22, d22, #1   \n"
            "vext.32    d24, d24, d24, #1   \n"
            "sub        %[remain], #4       \n"

            "store_2w_%=:                   \n"
            "cmp        %[remain], #2       \n"
            "blt        store_1w_%=         \n"
            "vst1.16    {d18[0]}, [%[y0]]!  \n"
            "vst1.16    {d20[0]}, [%[y1]]!  \n"
            "vst1.16    {d22[0]}, [%[y2]]!  \n"
            "vst1.16    {d24[0]}, [%[y3]]!  \n"
            "vext.16    d18, d18, d18, #1   \n"
            "vext.16    d20, d20, d20, #1   \n"
            "vext.16    d22, d22, d22, #1   \n"
            "vext.16    d24, d24, d24, #1   \n"
            "sub        %[remain], #2       \n"

            "store_1w_%=:"
            "cmp        %[remain], #1       \n"
            "blt        end_%=              \n"
            "vst1.8     {d18[0]}, [%[y0]]!  \n"
            "vst1.8     {d20[0]}, [%[y1]]!  \n"
            "vst1.8     {d22[0]}, [%[y2]]!  \n"
            "vst1.8     {d24[0]}, [%[y3]]!  \n"

            "end_%=:                        \n"
            : [x0] "+r"(x0), [x1] "+r"(x1), [x2] "+r"(x2), [x3] "+r"(x3),
              [y0] "+r"(y0), [y1] "+r"(y1), [y2] "+r"(y2), [y3] "+r"(y3),
              [loop] "+r"(loop), [remain] "+r"(remain)
            : [scale] "r"(scale)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12");
        asm volatile(
            "vdup.s8    d0, %[val]          \n"
            "cmp        %[pad_loop], #0     \n"
            "ble        store_pad_2w_%=     \n"
            "loop_pad_4w_%=:                \n"
            "vst1.32    {d0[0]}, [%[y0]]!   \n"
            "vst1.32    {d0[0]}, [%[y1]]!   \n"
            "vst1.32    {d0[0]}, [%[y2]]!   \n"
            "vst1.32    {d0[0]}, [%[y3]]!   \n"
            "subs       %[pad_loop], #1     \n"
            "bne        loop_pad_4w_%=      \n"

            "store_pad_2w_%=:               \n"
            "cmp        %[pad_remain], #2   \n"
            "blt        store_pad_1w_%=     \n"
            "vst1.16    {d0[0]}, [%[y0]]!   \n"
            "vst1.16    {d0[0]}, [%[y1]]!   \n"
            "vst1.16    {d0[0]}, [%[y2]]!   \n"
            "vst1.16    {d0[0]}, [%[y3]]!   \n"
            "sub        %[pad_remain], #2   \n"

            "store_pad_1w_%=:               \n"
            "cmp        %[pad_remain], #1   \n"
            "blt        end_%=              \n"
            "vst1.8    {d0[0]}, [%[y0]]!    \n"
            "vst1.8    {d0[0]}, [%[y1]]!    \n"
            "vst1.8    {d0[0]}, [%[y2]]!    \n"
            "vst1.8    {d0[0]}, [%[y3]]!    \n"
            "end_%=:                        \n"
            : [y0] "+r"(y0), [y1] "+r"(y1), [y2] "+r"(y2), [y3] "+r"(y3),
              [pad_loop] "+r"(pad_loop), [pad_remain] "+r"(pad_remain)
            : [val] "r"(padding_val)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12");
      }
    }
    for (int c = (channels & 0xFFFC); c < channels; ++c) {
      const float *input0 = x + (batch * channels + c) * input_spatial_size;
      size_t offset = (batch * channels + c) * output_spatial_size;
      for (int h = 0; h < 2; ++h) {
        int8_t *y0 =
            y + offset + h * ((input_h + paddings[0]) * output_w - paddings[1]);
        int loop = start >> 4;
        int remain = start & 0xF;
        asm volatile(
            "vdup.s8    q0,     %[val]      \n"
            "cmp        %[loop], #0         \n"
            "ble        start_remain_%=     \n"

            "store_16w_%=:                  \n"
            "vst1.32    {q0}, [%[y0]]!      \n"
            "subs       %[loop], #1         \n"
            "bne        store_16w_%=        \n"

            "start_remain_%=:               \n"
            "cmp        %[remain], #8       \n"
            "blt        store_4w_%=         \n"
            "vst1.32    {d0}, [%[y0]]!      \n"
            "sub        %[remain], #8       \n"

            "store_4w_%=:                   \n"
            "cmp        %[remain], #4       \n"
            "blt        store_2w_%=         \n"
            "vst1.32    {d0[0]}, [%[y0]]!   \n"
            "sub        %[remain], #4       \n"

            "store_2w_%=:                   \n"
            "cmp        %[remain], #4       \n"
            "blt        store_1w_%=         \n"
            "vst1.16    {d0[0]}, [%[y0]]!   \n"
            "sub        %[remain], #2       \n"

            "store_1w_%=:                   \n"
            "cmp        %[remain], #1       \n"
            "blt        end_%=              \n"
            "vst1.8     {d0[0]}, [%[y0]]!   \n"
            "end_%=:                        \n"
            : [y0] "+r"(y0), [loop] "+r"(loop), [remain] "+r"(remain)
            : [val] "r"(padding_val)
            : "cc", "memory", "q0");
      }
      // quantize valid area
      int8_t *y0 = y + offset + start;
      for (int h = 0; h < input_h; ++h) {
        const float *x0 = input0 + h * input_w;
        int loop = input_w >> 4;
        int remain = input_w & 0xF;
        int pad_loop = paddings[1] >> 1;  // (paddings[1] << 1) >> 2
        int pad_remain = (paddings[1] << 1) & 0x3;
        asm volatile(
            "vdup.f32   q0, %[scale]        \n"
            "cmp        %[loop], #0         \n"
            "ble        quantize_remain_%=  \n"

            "loop_quantize_%=:              \n"
            "vld1.32    {q1, q2}, [%[x0]]!  \n"
            "vmul.f32   q1, q1, q0          \n"
            "vmul.f32   q2, q2, q0          \n"
            "vcvt.s32.f32  q1, q1           \n"
            "vcvt.s32.f32  q2, q2           \n"
            "vmovn.s32  d2, q1              \n"
            "vmovn.s32  d3, q2              \n"
            "vmovn.s16  d18, q1             \n"
            "vld1.32    {q1, q2}, [%[x0]]!  \n"
            "vmul.f32   q1, q1, q0          \n"
            "vmul.f32   q2, q2, q0          \n"
            "vcvt.s32.f32  q1, q1           \n"
            "vcvt.s32.f32  q2, q2           \n"
            "vmovn.s32  d2, q1              \n"
            "vmovn.s32  d3, q2              \n"
            "vmovn.s16  d19, q1             \n"
            "vst1.32    {q9}, [%[y0]]!      \n"

            "subs       %[loop], #1         \n"
            "bne        loop_quantize_%=    \n"

            "quantize_remain_%=:            \n"
            "cmp        %[remain], #0       \n"
            "ble        start_pad_%=        \n"

            "vldm       %[x0], {d2-d9}      \n"
            "vmul.f32   q1, q1, q0          \n"
            "vmul.f32   q2, q2, q0          \n"
            "vcvt.s32.f32  q1, q1           \n"
            "vcvt.s32.f32  q2, q2           \n"
            "vmovn.s32  d2, q1              \n"
            "vmovn.s32  d3, q2              \n"
            "vmovn.s16  d18, q1             \n"
            "vmul.f32   q3, q3, q0          \n"
            "vmul.f32   q4, q4, q0          \n"
            "vcvt.s32.f32  q1, q3           \n"
            "vcvt.s32.f32  q2, q4           \n"
            "vmovn.s32  d2, q1              \n"
            "vmovn.s32  d3, q2              \n"
            "vmovn.s16  d19, q1             \n"

            "cmp        %[remain], #8       \n"
            "blt        store_4w_%=         \n"
            "vst1.32    {d18}, [%[y0]]!     \n"
            "vmov.32    d18, d19            \n"
            "sub        %[remain], #8       \n"

            "store_4w_%=:                   \n"
            "cmp        %[remain], #4       \n"
            "blt        store_2w_%=         \n"
            "vst1.32    {d18[0]}, [%[y0]]!  \n"
            "vext.32    d18, d18, d18, #1   \n"
            "sub        %[remain], #4       \n"

            "store_2w_%=:                   \n"
            "cmp        %[remain], #2       \n"
            "blt        store_1w_%=         \n"
            "vst1.16    {d18[0]}, [%[y0]]!  \n"
            "vext.16    d18, d18, d18, #1   \n"
            "sub        %[remain], #2       \n"

            "store_1w_%=:"
            "cmp        %[remain], #1       \n"
            "blt        start_pad_%=        \n"
            "vst1.8     {d18[0]}, [%[y0]]!  \n"

            "start_pad_%=:                  \n"
            "vdup.s8    d0, %[val]          \n"
            "cmp        %[pad_loop], #0     \n"
            "ble        pad_remain_%=       \n"
            "loop_pad_4w_%=:                \n"
            "vst1.32    {d0[0]}, [%[y0]]!   \n"
            "subs       %[pad_loop], #1     \n"
            "bne        loop_pad_4w_%=      \n"

            "pad_remain_%=:                 \n"
            "cmp        %[pad_remain], #2   \n"
            "blt        store_pad_1w_%=     \n"
            "vst1.16    {d0[0]}, [%[y0]]!   \n"
            "sub        %[pad_remain], #2   \n"

            "store_pad_1w_%=:               \n"
            "cmp        %[pad_remain], #1   \n"
            "blt        end_%=              \n"
            "vst1.8     {d0[0]}, [%[y0]]!   \n"
            "end_%=:                        \n"
            : [x0] "+r"(x0), [y0] "+r"(y0), [loop] "+r"(loop),
              [remain] "+r"(remain), [pad_loop] "+r"(pad_loop),
              [pad_remain] "+r"(pad_remain)
            : [scale] "r"(scale), [val] "r"(padding_val)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q9");
      }
    }
  }
}
#endif  // __aarch64__
#endif  // ARM_NEON

template <>
bool QuantizeKernel<CPU, float>::Init(QuantizeParam<CPU> *param) {
  return true;
}

template <>
void QuantizeKernel<CPU, float>::Compute(const QuantizeParam<CPU> &param) {
  const Tensor *input = param.input_;
  Tensor *output = param.output_;
  Tensor *output_scale = param.online_scale_;
  float max_abs = 0.f;
  if (param.is_static_) {
    max_abs = param.static_scale_;
  } else {
    max_abs = find_abs_max(input);
  }
  max_abs = std::max(max_abs, 1e-6f);
  // only support int8 currently
  float scale = 127 / max_abs;
  param.online_scale_->mutable_data<float>()[0] = max_abs;
  const auto &paddings = param.paddings_;
  // std::vector<int> paddings = {0, 0};
  // const auto padding_val = param.padding_val_;
  int8_t padding_val = 0;
  switch (param.round_type_) {
    case ROUND_NEAREST_TO_EVEN:
      quantize_round_to_even(input, scale, paddings, padding_val, output);
      break;
    case ROUND_NEAREST_TOWARDS_ZERO:
      quantize_round_to_zero(input, scale, paddings, padding_val, output);
      break;
    case ROUND_NEAREST_AWAY_ZERO:
      quantize_round_to_nearest(input, scale, paddings, padding_val, output);
      break;
    default:
      LOG(kLOG_ERROR) << "round type is not supported.";
      break;
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
