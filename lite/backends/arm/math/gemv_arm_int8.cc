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

#include "lite/backends/arm/math/gemv_arm_int8.h"
#include <arm_neon.h>
#include "lite/backends/arm/math/saturate.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename dtype>
inline void write_gemv_out(const int* in,
                           dtype* out,
                           const float* scale,
                           const float* bias,
                           int size,
                           bool flag_act,
                           lite_api::ActivationType act,
                           float six,
                           float alpha);

template <>
inline void write_gemv_out(const int* in,
                           float* out,
                           const float* scale,
                           const float* bias,
                           int size,
                           bool flag_act,
                           lite_api::ActivationType act,
                           float six,
                           float alpha) {
  int i = 0;
  float32x4_t vzero = vdupq_n_f32(0.f);
  for (; i < size - 7; i += 8) {
    float32x4_t vout0 = bias ? vld1q_f32(bias) : vdupq_n_f32(0.f);
    float32x4_t vout1 = bias ? vld1q_f32(bias + 4) : vdupq_n_f32(0.f);
    int32x4_t vin0 = vld1q_s32(in);
    int32x4_t vin1 = vld1q_s32(in + 4);
    float32x4_t vscale0 = vld1q_f32(scale);
    float32x4_t vscale1 = vld1q_f32(scale + 4);
    float32x4_t vinf0 = vcvtq_f32_s32(vin0);
    float32x4_t vinf1 = vcvtq_f32_s32(vin1);
    vout0 = vmlaq_f32(vout0, vinf0, vscale0);
    vout1 = vmlaq_f32(vout1, vinf1, vscale1);
    if (flag_act) {
      if (act == lite_api::ActivationType::kRelu) {
        vout0 = vmaxq_f32(vout0, vzero);
        vout1 = vmaxq_f32(vout1, vzero);
      } else if (act == lite_api::ActivationType::kRelu6) {
        float32x4_t vsix = vdupq_n_f32(six);
        vout0 = vmaxq_f32(vout0, vzero);
        vout1 = vmaxq_f32(vout1, vzero);
        vout0 = vminq_f32(vout0, vsix);
        vout1 = vminq_f32(vout1, vsix);
      } else if (act == lite_api::ActivationType::kLeakyRelu) {
        float32x4_t valpha = vdupq_n_f32(alpha);
        uint32x4_t maska = vcgeq_f32(vout0, vzero);
        uint32x4_t maskb = vcgeq_f32(vout1, vzero);
        float32x4_t suma = vmulq_f32(vout0, valpha);
        float32x4_t sumb = vmulq_f32(vout1, valpha);
        vout0 = vbslq_f32(maska, vout0, suma);
        vout1 = vbslq_f32(maskb, vout1, sumb);
      }
    }
    vst1q_f32(out, vout0);
    vst1q_f32(out + 4, vout1);
    bias += 8;
    in += 8;
    out += 8;
    scale += 8;
  }
  for (; i < size; ++i) {
    out[0] = *(in++) * *(scale)++;
    out[0] += bias ? *(bias++) : 0.f;
    if (flag_act) {
      if (act == lite_api::ActivationType::kRelu) {
        out[0] = out[0] > 0.f ? out[0] : 0.f;
      } else if (act == lite_api::ActivationType::kRelu6) {
        out[0] = out[0] > 0.f ? (out[0] > six ? six : out[0]) : 0.f;
      } else if (act == lite_api::ActivationType::kLeakyRelu) {
        out[0] = out[0] > 0.f ? out[0] : out[0] * alpha;
      }
    }
    out++;
  }
}

template <>
inline void write_gemv_out(const int* in,
                           signed char* out,
                           const float* scale,
                           const float* bias,
                           int size,
                           bool flag_act,
                           lite_api::ActivationType act,
                           float six,
                           float alpha) {
  if (bias) {
    for (int i = 0; i < size; ++i) {
      float tmp = *(in++) * *(scale++) + *(bias++);
      if (flag_act) {
        if (act == lite_api::ActivationType::kRelu) {
          tmp = tmp > 0.f ? tmp : 0.f;
        } else if (act == lite_api::ActivationType::kRelu6) {
          tmp = tmp > 0.f ? (tmp > six ? six : tmp) : 0.f;
        } else if (act == lite_api::ActivationType::kLeakyRelu) {
          tmp = tmp > 0.f ? tmp : (tmp * alpha);
        }
      }
      out[0] = saturate_cast<signed char>(roundf(tmp));
      out[0] = out[0] < -127 ? -127 : out[0];  // -127 - 127
      out++;
    }
  } else {
    for (int i = 0; i < size; ++i) {
      float tmp = *(in++) * *(scale++);
      if (flag_act) {
        if (act == lite_api::ActivationType::kRelu) {
          tmp = tmp > 0.f ? tmp : 0.f;
        } else if (act == lite_api::ActivationType::kRelu6) {
          tmp = tmp > 0.f ? (tmp > six ? six : tmp) : 0.f;
        } else if (act == lite_api::ActivationType::kLeakyRelu) {
          tmp = tmp > 0.f ? tmp : tmp * alpha;
        }
      }
      out[0] = saturate_cast<signed char>(roundf(tmp));
      out[0] = out[0] < -127 ? -127 : out[0];  // -127 - 127
      out++;
    }
  }
}

template <typename dtype>
bool gemv_int8_trans_oth(const int8_t* A,
                         const int8_t* x,
                         dtype* y,
                         int M,
                         int N,
                         const float* scale,
                         bool is_bias,
                         const float* bias,
                         bool flag_act,
                         lite_api::ActivationType act,
                         float six,
                         float alpha) {
  dtype* data_out = y;
  const int8_t* data_in = A;
  const int8_t* weights_ptr = x;
  int out_cnt = M >> 4;
  int out_remain = M & 15;
  int zero_ptr[M];  // NOLINT
  memset(zero_ptr, 0, sizeof(int) * M);
  float zerobuf[M];  // NOLINT
  memset(zerobuf, 0, sizeof(float) * M);
  const float* bias_ptr = is_bias ? bias : zerobuf;

#ifdef __aarch64__
  int cnt = N >> 3;
  int tail = N & 7;
  int cnt_4 = tail >> 2;
  int tail_4 = tail & 3;
  int stride_in = M << 3;
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const int8_t* in_ptr0 = data_in;
    const int8_t* in_ptr1 = in_ptr0 + M;
    const int8_t* in_ptr2 = in_ptr1 + M;
    const int8_t* in_ptr3 = in_ptr2 + M;
    const int8_t* in_ptr4 = in_ptr3 + M;
    const int8_t* in_ptr5 = in_ptr4 + M;
    const int8_t* in_ptr6 = in_ptr5 + M;
    const int8_t* in_ptr7 = in_ptr6 + M;
    const int8_t* wei_ptr = weights_ptr;
    int32_t* out_ptr = zero_ptr;
    int cnt_col = out_cnt;
    asm volatile(
        "prfm  pldl1keep, [%[wc]]        \n"
        "prfm  pldl1keep, [%[in_ptr0]]   \n"
        "prfm  pldl1keep, [%[in_ptr1]]   \n"
        "prfm  pldl1keep, [%[in_ptr2]]   \n"
        "prfm  pldl1keep, [%[in_ptr3]]   \n"
        "prfm  pldl1keep, [%[in_ptr4]]   \n"
        "prfm  pldl1keep, [%[in_ptr5]]   \n"
        "prfm  pldl1keep, [%[in_ptr6]]   \n"
        "prfm  pldl1keep, [%[in_ptr7]]   \n"
        "cmp %w[cnt], #1\n"
        "ldr q8, [%[wc]]\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "ldr q10, [%[in_ptr1]], #0x10\n"
        "ldr q11, [%[in_ptr2]], #0x10\n"
        "ldr q12, [%[in_ptr3]], #0x10\n"
        "sxtl  v0.8h, v8.8b\n"
        "blt 2f\n"
        "1: \n"
        "ldr q13, [%[out_ptr]]\n"
        "sxtl  v1.8h, v9.8b\n"
        "sxtl2 v2.8h, v9.16b\n"
        "ldr q14, [%[out_ptr], #0x10]\n"
        "sxtl  v3.8h, v10.8b\n"
        "sxtl2 v4.8h, v10.16b\n"
        "ldr q15, [%[out_ptr], #0x20]\n"
        "sxtl  v5.8h, v11.8b\n"
        "sxtl2 v6.8h, v11.16b\n"
        "ldr q16, [%[out_ptr], #0x30]\n"
        "ldr q9, [%[in_ptr4]], #0x10\n"
        // r0
        "smlal v13.4s, v1.4h, v0.h[0]\n"
        "smlal2 v14.4s, v1.8h, v0.h[0]\n"
        "sxtl  v7.8h, v12.8b\n"
        "sxtl2 v8.8h, v12.16b\n"
        "ldr q10, [%[in_ptr5]], #0x10\n"
        "smlal v15.4s, v2.4h, v0.h[0]\n"
        "smlal2 v16.4s, v2.8h, v0.h[0]\n"
        "sxtl  v17.8h, v9.8b\n"
        "sxtl2 v18.8h, v9.16b\n"
        "ldr q11, [%[in_ptr6]], #0x10\n"
        // r1
        "smlal v13.4s, v3.4h, v0.h[1]\n"
        "smlal2 v14.4s, v3.8h, v0.h[1]\n"
        "sxtl  v19.8h, v10.8b\n"
        "sxtl2 v20.8h, v10.16b\n"
        "ldr q12, [%[in_ptr7]], #0x10\n"
        "smlal v15.4s, v4.4h, v0.h[1]\n"
        "smlal2 v16.4s, v4.8h, v0.h[1]\n"
        "sxtl  v1.8h, v11.8b\n"
        "sxtl2 v2.8h, v11.16b\n"
        // r2
        "smlal v13.4s, v5.4h, v0.h[2]\n"
        "smlal2 v14.4s, v5.8h, v0.h[2]\n"
        "sxtl  v3.8h, v12.8b\n"
        "sxtl2 v4.8h, v12.16b\n"
        "smlal v15.4s, v6.4h, v0.h[2]\n"
        "smlal2 v16.4s, v6.8h, v0.h[2]\n"
        // r3
        "smlal v13.4s, v7.4h, v0.h[3]\n"
        "smlal2 v14.4s, v7.8h, v0.h[3]\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "smlal v15.4s, v8.4h, v0.h[3]\n"
        "smlal2 v16.4s, v8.8h, v0.h[3]\n"
        // r4
        "smlal v13.4s, v17.4h, v0.h[4]\n"
        "smlal2 v14.4s, v17.8h, v0.h[4]\n"
        "ldr q10, [%[in_ptr1]], #0x10\n"
        "smlal v15.4s, v18.4h, v0.h[4]\n"
        "smlal2 v16.4s, v18.8h, v0.h[4]\n"
        // r5
        "smlal v13.4s, v19.4h, v0.h[5]\n"
        "smlal2 v14.4s, v19.8h, v0.h[5]\n"
        "ldr q11, [%[in_ptr2]], #0x10\n"
        "smlal v15.4s, v20.4h, v0.h[5]\n"
        "smlal2 v16.4s, v20.8h, v0.h[5]\n"
        // r6
        "smlal v13.4s, v1.4h, v0.h[6]\n"
        "smlal2 v14.4s, v1.8h, v0.h[6]\n"
        "ldr q12, [%[in_ptr3]], #0x10\n"
        "smlal v15.4s, v2.4h, v0.h[6]\n"
        "smlal2 v16.4s, v2.8h, v0.h[6]\n"
        // r7
        "smlal v13.4s, v3.4h, v0.h[7]\n"
        "smlal2 v14.4s, v3.8h, v0.h[7]\n"
        "smlal v15.4s, v4.4h, v0.h[7]\n"
        "smlal2 v16.4s, v4.8h, v0.h[7]\n"
        "subs %w[cnt], %w[cnt], #1\n"
        "str q13, [%[out_ptr]], #0x10\n"
        "str q14, [%[out_ptr]], #0x10\n"
        "str q15, [%[out_ptr]], #0x10\n"
        "str q16, [%[out_ptr]], #0x10\n"
        "bne 1b\n"
        "2: \n"
        : [out_ptr] "+r"(out_ptr),
          [in_ptr0] "+r"(in_ptr0),
          [in_ptr1] "+r"(in_ptr1),
          [in_ptr2] "+r"(in_ptr2),
          [in_ptr3] "+r"(in_ptr3),
          [in_ptr4] "+r"(in_ptr4),
          [in_ptr5] "+r"(in_ptr5),
          [in_ptr6] "+r"(in_ptr6),
          [in_ptr7] "+r"(in_ptr7),
          [cnt] "+r"(cnt_col)
        : [wc] "r"(wei_ptr)
        : "cc",
          "memory",
          "v0",
          "v1",
          "v2",
          "v3",
          "v4",
          "v5",
          "v6",
          "v7",
          "v8",
          "v9",
          "v10",
          "v11",
          "v12",
          "v13",
          "v14",
          "v15",
          "v16",
          "v17",
          "v18",
          "v19",
          "v20");
    in_ptr0 -= 16;
    in_ptr1 -= 16;
    in_ptr2 -= 16;
    in_ptr3 -= 16;
    for (int j = 0; j < out_remain; j++) {
      *out_ptr += *in_ptr0++ * wei_ptr[0];
      *out_ptr += *in_ptr1++ * wei_ptr[1];
      *out_ptr += *in_ptr2++ * wei_ptr[2];
      *out_ptr += *in_ptr3++ * wei_ptr[3];
      *out_ptr += *in_ptr4++ * wei_ptr[4];
      *out_ptr += *in_ptr5++ * wei_ptr[5];
      *out_ptr += *in_ptr6++ * wei_ptr[6];
      *out_ptr += *in_ptr7++ * wei_ptr[7];
      out_ptr++;
    }
    data_in += stride_in;
    weights_ptr += 8;
  }
  if (cnt_4) {
    const int8_t* in_ptr0 = data_in;
    const int8_t* in_ptr1 = in_ptr0 + M;
    const int8_t* in_ptr2 = in_ptr1 + M;
    const int8_t* in_ptr3 = in_ptr2 + M;
    int32_t* out_ptr = zero_ptr;
    int cnt_col = out_cnt;
    const int8_t* wei_ptr = weights_ptr;
    asm volatile(
        "prfm  pldl1keep, [%[wc]]        \n"
        "prfm  pldl1keep, [%[in_ptr0]]   \n"
        "prfm  pldl1keep, [%[in_ptr1]]   \n"
        "prfm  pldl1keep, [%[in_ptr2]]   \n"
        "prfm  pldl1keep, [%[in_ptr3]]   \n"
        "cmp %w[cnt], #1\n"
        "ldr q8, [%[wc]]\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "ldr q10, [%[in_ptr1]], #0x10\n"
        "ldr q11, [%[in_ptr2]], #0x10\n"
        "sxtl  v0.8h, v8.8b\n"
        "ldr q12, [%[in_ptr3]], #0x10\n"
        "blt 2f\n"
        "1: \n"
        "ldr q13, [%[out_ptr]]\n"
        "sxtl  v1.8h, v9.8b\n"
        "sxtl2 v2.8h, v9.16b\n"
        "ldr q14, [%[out_ptr], #0x10]\n"
        "sxtl  v3.8h, v10.8b\n"
        "sxtl2 v4.8h, v10.16b\n"
        "ldr q15, [%[out_ptr], #0x20]\n"
        "sxtl  v5.8h, v11.8b\n"
        "sxtl2 v6.8h, v11.16b\n"
        "ldr q16, [%[out_ptr], #0x30]\n"
        // r0
        "smlal v13.4s, v1.4h, v0.h[0]\n"
        "smlal2 v14.4s, v1.8h, v0.h[0]\n"
        "sxtl  v7.8h, v12.8b\n"
        "sxtl2 v8.8h, v12.16b\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "smlal v15.4s, v2.4h, v0.h[0]\n"
        "smlal2 v16.4s, v2.8h, v0.h[0]\n"
        // r1
        "smlal v13.4s, v3.4h, v0.h[1]\n"
        "smlal2 v14.4s, v3.8h, v0.h[1]\n"
        "ldr q10, [%[in_ptr1]], #0x10\n"
        "smlal v15.4s, v4.4h, v0.h[1]\n"
        "smlal2 v16.4s, v4.8h, v0.h[1]\n"
        // r2
        "smlal v13.4s, v5.4h, v0.h[2]\n"
        "smlal2 v14.4s, v5.8h, v0.h[2]\n"
        "ldr q11, [%[in_ptr2]], #0x10\n"
        "smlal v15.4s, v6.4h, v0.h[2]\n"
        "smlal2 v16.4s, v6.8h, v0.h[2]\n"
        // r3
        "smlal v13.4s, v7.4h, v0.h[3]\n"
        "smlal2 v14.4s, v7.8h, v0.h[3]\n"
        "ldr q12, [%[in_ptr3]], #0x10\n"
        "smlal v15.4s, v8.4h, v0.h[3]\n"
        "smlal2 v16.4s, v8.8h, v0.h[3]\n"
        "subs %w[cnt], %w[cnt], #1\n"
        "str q13, [%[out_ptr]], #0x10\n"
        "str q14, [%[out_ptr]], #0x10\n"
        "str q15, [%[out_ptr]], #0x10\n"
        "str q16, [%[out_ptr]], #0x10\n"
        "bne 1b\n"
        "2: \n"
        : [out_ptr] "+r"(out_ptr),
          [in_ptr0] "+r"(in_ptr0),
          [in_ptr1] "+r"(in_ptr1),
          [in_ptr2] "+r"(in_ptr2),
          [in_ptr3] "+r"(in_ptr3),
          [cnt] "+r"(cnt_col)
        : [wc] "r"(wei_ptr)
        : "cc",
          "memory",
          "v0",
          "v1",
          "v2",
          "v3",
          "v4",
          "v5",
          "v6",
          "v7",
          "v8",
          "v9",
          "v10",
          "v11",
          "v12",
          "v13",
          "v14",
          "v15",
          "v16");
    in_ptr0 -= 16;
    in_ptr1 -= 16;
    in_ptr2 -= 16;
    in_ptr3 -= 16;
    for (int j = 0; j < out_remain; j++) {
      *out_ptr += *in_ptr0++ * wei_ptr[0];
      *out_ptr += *in_ptr1++ * wei_ptr[1];
      *out_ptr += *in_ptr2++ * wei_ptr[2];
      *out_ptr += *in_ptr3++ * wei_ptr[3];
      out_ptr++;
    }
    data_in += 4 * M;
    weights_ptr += 4;
  }
  for (int i = 0; i < tail_4; i++) {
    const int8_t* in_ptr = data_in;
    const int8_t* wei_ptr = weights_ptr;
    int32_t* out_ptr = zero_ptr;
    int cnt_col = out_cnt;
    asm volatile(
        "prfm  pldl1keep, [%[wc]]        \n"
        "prfm  pldl1keep, [%[out_ptr]]   \n"
        "prfm  pldl1keep, [%[in_ptr0]]   \n"
        "cmp %w[cnt], #1\n"
        "ldr q8, [%[wc]]\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "ldr q13, [%[out_ptr]]\n"
        "ldr q14, [%[out_ptr], #0x10]\n"
        "sxtl  v0.8h, v8.8b\n"
        "blt 2f\n"
        "1: \n"
        "sxtl  v1.8h, v9.8b\n"
        "sxtl2 v2.8h, v9.16b\n"
        "ldr q15, [%[out_ptr], #0x20]\n"
        "ldr q16, [%[out_ptr], #0x30]\n"
        // r0
        "smlal v13.4s, v1.4h, v0.h[0]\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "smlal2 v14.4s, v1.8h, v0.h[0]\n"
        "smlal v15.4s, v2.4h, v0.h[0]\n"
        "smlal2 v16.4s, v2.8h, v0.h[0]\n"

        "subs %w[cnt], %w[cnt], #1\n"
        "str q13, [%[out_ptr]], #0x10\n"
        "str q14, [%[out_ptr]], #0x10\n"
        "str q15, [%[out_ptr]], #0x10\n"
        "str q16, [%[out_ptr]], #0x10\n"
        "ldr q13, [%[out_ptr]]\n"
        "ldr q14, [%[out_ptr], #0x10]\n"
        "bne 1b\n"
        "2: \n"
        : [out_ptr] "+r"(out_ptr), [in_ptr0] "+r"(in_ptr), [cnt] "+r"(cnt_col)
        : [wc] "r"(wei_ptr)
        : "cc",
          "memory",
          "v0",
          "v1",
          "v2",
          "v3",
          "v4",
          "v5",
          "v6",
          "v7",
          "v8",
          "v9",
          "v10",
          "v11",
          "v12",
          "v13",
          "v14",
          "v15",
          "v16");
    in_ptr -= 16;
    for (int j = 0; j < out_remain; j++) {
      *out_ptr += *in_ptr++ * wei_ptr[0];
      out_ptr++;
    }
    data_in += M;
    weights_ptr++;
  }
#else
  int cnt = N >> 2;
  int tail = N & 3;
  int stride_in = M << 2;
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const int8_t* in_ptr0 = data_in;
    const int8_t* in_ptr1 = in_ptr0 + M;
    const int8_t* in_ptr2 = in_ptr1 + M;
    const int8_t* in_ptr3 = in_ptr2 + M;
    const int8_t* wei_ptr = weights_ptr;
    int32_t* out_ptr = zero_ptr;
    int cnt_col = out_cnt;
    asm volatile(
        "pld [%[wc]]        \n"
        "pld [%[in_ptr0]]   \n"
        "pld [%[in_ptr1]]   \n"
        "pld [%[in_ptr2]]   \n"
        "pld [%[in_ptr3]]   \n"
        "pld [%[out_ptr]]   \n"
        "cmp %[cnt], #1\n"
        "vld1.8 {d22}, [%[wc]]\n"
        "blt 2f\n"
        "1: \n"
        "vld1.8 {d10-d11}, [%[in_ptr0]]!\n"
        "vmovl.s8 q0, d22\n"
        "vld1.8 {d12-d13}, [%[in_ptr1]]!\n"
        "vld1.32 {d24-d25}, [%[out_ptr]]\n"
        "vld1.8 {d14-d15}, [%[in_ptr2]]!\n"
        "vldr d26, [%[out_ptr], #0x10]\n"
        "vld1.8 {d16-d17}, [%[in_ptr3]]!\n"
        "vmovl.s8 q1, d10\n"
        "vmovl.s8 q2, d11\n"
        "vldr d27, [%[out_ptr], #0x18]\n"
        "vmovl.s8 q3, d12\n"
        "vmovl.s8 q4, d13\n"
        "vldr d28, [%[out_ptr], #0x20]\n"
        "vldr d29, [%[out_ptr], #0x28]\n"
        "vmovl.s8 q9, d14\n"
        "vmovl.s8 q10, d15\n"
        "vldr d30, [%[out_ptr], #0x30]\n"
        "vldr d31, [%[out_ptr], #0x38]\n"
        // r0
        "vmlal.s16 q12, d2, d0[0]\n"
        "vmlal.s16 q13, d3, d0[0]\n"
        "vmovl.s8 q5, d16\n"
        "vmlal.s16 q14, d4, d0[0]\n"
        "vmlal.s16 q15, d5, d0[0]\n"
        "vmovl.s8 q6, d17\n"
        // r1
        "vmlal.s16 q12, d6, d0[1]\n"
        "vmlal.s16 q13, d7, d0[1]\n"
        "vmlal.s16 q14, d8, d0[1]\n"
        "vmlal.s16 q15, d9, d0[1]\n"
        // r2
        "vmlal.s16 q12, d18, d0[2]\n"
        "vmlal.s16 q13, d19, d0[2]\n"
        "vmlal.s16 q14, d20, d0[2]\n"
        "vmlal.s16 q15, d21, d0[2]\n"
        // r3
        "vmlal.s16 q12, d10, d0[3]\n"
        "vmlal.s16 q13, d11, d0[3]\n"
        "vmlal.s16 q14, d12, d0[3]\n"
        "vmlal.s16 q15, d13, d0[3]\n"

        "subs %[cnt], #1\n"
        "vst1.32 {d24-d25}, [%[out_ptr]]!\n"
        "vst1.32 {d26-d27}, [%[out_ptr]]!\n"
        "vst1.32 {d28-d29}, [%[out_ptr]]!\n"
        "vst1.32 {d30-d31}, [%[out_ptr]]!\n"
        "bne 1b\n"
        "2: \n"
        : [out_ptr] "+r"(out_ptr),
          [in_ptr0] "+r"(in_ptr0),
          [in_ptr1] "+r"(in_ptr1),
          [in_ptr2] "+r"(in_ptr2),
          [in_ptr3] "+r"(in_ptr3),
          [cnt] "+r"(cnt_col)
        : [wc] "r"(wei_ptr)
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
          "q11",
          "q12",
          "q13",
          "q14",
          "q15");
    for (int j = 0; j < out_remain; j++) {
      *out_ptr += *in_ptr0++ * wei_ptr[0];
      *out_ptr += *in_ptr1++ * wei_ptr[1];
      *out_ptr += *in_ptr2++ * wei_ptr[2];
      *out_ptr += *in_ptr3++ * wei_ptr[3];
      out_ptr++;
    }
    data_in += stride_in;
    weights_ptr += 4;
  }
  for (int i = 0; i < tail; i++) {
    const int8_t* in_ptr = data_in;
    const int8_t* wei_ptr = weights_ptr;
    int32_t* out_ptr = zero_ptr;
    int cnt_col = out_cnt;
    asm volatile(
        "pld [%[wc]]        \n"
        "pld [%[in_ptr0]]   \n"
        "pld [%[out_ptr]]   \n"
        "cmp %[cnt], #1\n"
        "vld1.8 {d8}, [%[wc]]\n"
        "vld1.8 {d10-d11}, [%[in_ptr0]]!\n"
        "vld1.32 {d24-d25}, [%[out_ptr]]\n"
        "vldr d26, [%[out_ptr], #0x10]\n"
        "vmovl.s8 q0, d8\n"
        "blt 2f\n"
        "1: \n"
        "vldr d27, [%[out_ptr], #0x18]\n"
        "vmovl.s8 q1, d10\n"
        "vldr d28, [%[out_ptr], #0x20]\n"
        "vmovl.s8 q2, d11\n"
        "vldr d29, [%[out_ptr], #0x28]\n"
        // r0
        "vmlal.s16 q12, d2, d0[0]\n"
        "vldr d30, [%[out_ptr], #0x30]\n"
        "vldr d31, [%[out_ptr], #0x38]\n"
        "vmlal.s16 q13, d3, d0[0]\n"
        "vld1.8 {d10-d11}, [%[in_ptr0]]!\n"
        "vmlal.s16 q14, d4, d0[0]\n"
        "vmlal.s16 q15, d5, d0[0]\n"

        "subs %[cnt], #1\n"
        "vst1.32 {d24-d25}, [%[out_ptr]]!\n"
        "vst1.32 {d26-d27}, [%[out_ptr]]!\n"
        "vst1.32 {d28-d29}, [%[out_ptr]]!\n"
        "vst1.32 {d30-d31}, [%[out_ptr]]!\n"
        "vld1.32 {d24-d25}, [%[out_ptr]]\n"
        "vldr d26, [%[out_ptr], #0x10]\n"
        "bne 1b\n"
        "2: \n"
        : [out_ptr] "+r"(out_ptr), [in_ptr0] "+r"(in_ptr), [cnt] "+r"(cnt_col)
        : [wc] "r"(wei_ptr)
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
          "q12",
          "q13",
          "q14",
          "q15");
    in_ptr -= 16;
    for (int j = 0; j < out_remain; j++) {
      *out_ptr += *in_ptr++ * wei_ptr[0];
      out_ptr++;
    }
    data_in += M;
    weights_ptr++;
  }
#endif
  // write output
  write_gemv_out(zero_ptr, y, scale, bias_ptr, M, flag_act, act, six, alpha);
  return true;
}

template <typename dtype>
bool gemv_int8_oth(const int8_t* A,
                   const int8_t* x,
                   dtype* y,
                   bool transA,
                   int M,
                   int N,
                   const float* scale,
                   bool is_bias,
                   const float* bias,
                   bool flag_act,
                   lite_api::ActivationType act,
                   float six,
                   float alpha) {
  if (transA) {
    LOG(ERROR) << "ERROR: sgemv, transA is not supported now";
    return false;
  }
  dtype* data_out = y;
  const int8_t* data_in = x;
  const int8_t* weights_ptr = A;
  int cnt = N >> 4;
  int tail = N & 15;

#ifdef __aarch64__
  int out_cnt = M >> 3;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    dtype* out_ptr = data_out + out_idx;
    const float* scale_ptr = scale + out_idx;
    int ptr_out[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = weights_ptr + (N * out_idx);
    const int8_t* ptr_w1 = ptr_w0 + N;
    const int8_t* ptr_w2 = ptr_w1 + N;
    const int8_t* ptr_w3 = ptr_w2 + N;
    const int8_t* ptr_w4 = ptr_w3 + N;
    const int8_t* ptr_w5 = ptr_w4 + N;
    const int8_t* ptr_w6 = ptr_w5 + N;
    const int8_t* ptr_w7 = ptr_w6 + N;
    auto bias_ptr = is_bias ? bias + out_idx : nullptr;
    int cnt_loop = cnt;
    asm volatile(
        "prfm  pldl1keep, [%[in]]           \n" /* preload din */
        "prfm  pldl1keep, [%[w0]]   \n"         /* preload w0 */
        "prfm  pldl1keep, [%[w1]]   \n"         /* preload w1 */
        "prfm  pldl1keep, [%[w2]]   \n"         /* preload w2 */
        "prfm  pldl1keep, [%[w3]]   \n"         /* preload w3 */
        "prfm  pldl1keep, [%[w4]]   \n"         /* preload w4 */
        "prfm  pldl1keep, [%[w5]]   \n"         /* preload w5 */
        "prfm  pldl1keep, [%[w6]]   \n"         /* preload w6 */
        "prfm  pldl1keep, [%[w7]]   \n"         /* preload w7 */
        "movi   v0.4s,  #0          \n"         /* set out0 to 0 */
        "movi   v1.4s,  #0          \n"         /* set out1 to 0 */
        "movi   v2.4s,  #0          \n"         /* set out2 to 0 */
        "movi   v3.4s,  #0          \n"         /* set out3 to 0 */
        "movi   v4.4s,  #0          \n"         /* set out4 to 0 */
        "movi   v5.4s,  #0          \n"         /* set out5 to 0 */
        "movi   v6.4s,  #0          \n"         /* set out6 to 0 */
        "movi   v7.4s,  #0          \n"         /* set out7 to 0 */
        /* check main loop */
        "cmp %w[cnt], #1            \n" /* check whether has main loop */
        "blt  2f                    \n" /* jump to tail */
        /* main loop */
        "1:                         \n"  /* main loop */
        "ldr    q8,     [%[in]], #16 \n" /* load input, 16 int8 */
        "ldr    q9,     [%[w0]], #16 \n" /* load w0, 16 int8 */
        "ldr    q10,    [%[w1]], #16 \n" /* load w1, 16 int8 */
        "ldr    q11,    [%[w2]], #16 \n" /* load w2, 16 int8 */
        "ldr    q12,    [%[w3]], #16 \n" /* load w3, 16 int8 */
        "ldr    q13,    [%[w4]], #16 \n" /* load w4, 16 int8 */
        "ldr    q14,    [%[w5]], #16 \n" /* load w5, 16 int8 */
        "ldr    q15,    [%[w6]], #16 \n" /* load w6, 16 int8 */
        "ldr    q16,    [%[w7]], #16 \n" /* load w7, 16 int8 */
        /* mul, lower 8 int8 * int8 = int16 */
        "smull  v18.8h, v8.8b, v9.8b \n" /* mul in * w0, low, 8 int8 */
        "smull  v19.8h, v8.8b, v10.8b\n" /* mul in * w1, low, 8 int8 */
        "smull  v20.8h, v8.8b, v11.8b\n" /* mul in * w2, low, 8 int8 */
        "smull  v21.8h, v8.8b, v12.8b\n" /* mul in * w3, low, 8 int8 */
        "smull  v22.8h, v8.8b, v13.8b\n" /* mul in * w4, low, 8 int8 */
        "smull  v23.8h, v8.8b, v14.8b\n" /* mul in * w5, low, 8 int8 */
        "smull  v24.8h, v8.8b, v15.8b\n" /* mul in * w6, low, 8 int8 */
        "smull  v25.8h, v8.8b, v16.8b\n" /* mul in * w7, low, 8 int8 */
        /* mul, higher 8 int8 * int8 + int16 = int16 */
        "smlal2 v18.8h,v8.16b,v9.16b \n" /* mul in * w0, high, 8 int8 */
        "smlal2 v19.8h,v8.16b,v10.16b\n" /* mul in * w1, high, 8 int8 */
        "smlal2 v20.8h,v8.16b,v11.16b\n" /* mul in * w2, high, 8 int8 */
        "smlal2 v21.8h,v8.16b,v12.16b\n" /* mul in * w2, high, 8 int8 */
        "smlal2 v22.8h,v8.16b,v13.16b\n" /* mul in * w2, high, 8 int8 */
        "smlal2 v23.8h,v8.16b,v14.16b\n" /* mul in * w2, high, 8 int8 */
        "smlal2 v24.8h,v8.16b,v15.16b\n" /* mul in * w2, high, 8 int8 */
        "smlal2 v25.8h,v8.16b,v16.16b\n" /* mul in * w2, high, 8 int8 */
        "subs %w[cnt], %w[cnt], #1   \n" /* sub main loop count */
        /* add int16 to int32 */
        "sadalp v0.4s, v18.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v1.4s, v19.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v2.4s, v20.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v3.4s, v21.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v4.4s, v22.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v5.4s, v23.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v6.4s, v24.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "sadalp v7.4s, v25.8h \n"        /* pair acc, 8 int16 -> 4 int32 */
        "bne 1b                      \n" /* jump to main loop */
        /* pair add to final result */
        "2:                          \n" /* reduce to scale */
        "addp v8.4s , v0.4s , v1.4s  \n" /* pair add to 4 int32*/
        "addp v9.4s , v2.4s , v3.4s  \n" /* pair add to 4 int32*/
        "addp v10.4s, v4.4s , v5.4s  \n" /* pair add to 4 int32*/
        "addp v11.4s, v6.4s , v7.4s  \n" /* pair add to 4 int32*/

        "addp v12.4s, v8.4s , v9.4s  \n" /* pair add to 4 int32*/
        "addp v13.4s, v10.4s, v11.4s \n" /* pair add to 4 int32*/

        /* write to output */
        "stp q12, q13, [%[out]]     \n" /* save result */
        : [in] "+r"(ptr_in),
          [w0] "+r"(ptr_w0),
          [w1] "+r"(ptr_w1),
          [w2] "+r"(ptr_w2),
          [w3] "+r"(ptr_w3),
          [w4] "+r"(ptr_w4),
          [w5] "+r"(ptr_w5),
          [w6] "+r"(ptr_w6),
          [w7] "+r"(ptr_w7),
          [cnt] "+r"(cnt_loop)
        : [out] "r"(ptr_out)
        : "cc",
          "memory",
          "v0",
          "v1",
          "v2",
          "v3",
          "v4",
          "v5",
          "v6",
          "v7",
          "v8",
          "v9",
          "v10",
          "v11",
          "v12",
          "v13",
          "v14",
          "v15",
          "v16",
          "v17",
          "v18",
          "v19",
          "v20",
          "v21",
          "v22",
          "v23",
          "v24",
          "v25");
    for (int i = 0; i < tail; ++i) {
      ptr_out[0] += ptr_in[i] * ptr_w0[i];
      ptr_out[1] += ptr_in[i] * ptr_w1[i];
      ptr_out[2] += ptr_in[i] * ptr_w2[i];
      ptr_out[3] += ptr_in[i] * ptr_w3[i];
      ptr_out[4] += ptr_in[i] * ptr_w4[i];
      ptr_out[5] += ptr_in[i] * ptr_w5[i];
      ptr_out[6] += ptr_in[i] * ptr_w6[i];
      ptr_out[7] += ptr_in[i] * ptr_w7[i];
    }

    write_gemv_out(
        ptr_out, out_ptr, scale_ptr, bias_ptr, 8, flag_act, act, six, alpha);
  }

//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 8; j < M; j++) {
    // int *ptr_out = data_out + j;
    dtype* out_ptr = data_out + j;
    const float* scale_ptr = scale + j;
    int ptr_out[1] = {0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    auto bias_ptr = is_bias ? bias + j : nullptr;
    asm volatile(
        "prfm  pldl1keep, [%[in]]               \n" /* preload din */
        "prfm  pldl1keep, [%[w0]]       \n"         /* preload w0 */
        "movi   v0.4s,  #0              \n"         /* set out0 to 0 */
        /* check main loop */
        "cmp %w[cnt], #1                \n" /* check whether has main loop */
        "blt  2f                        \n" /* jump to tail */
        /* main loop */
        "1:                             \n" /* main loop */
        "ldr    q8,     [%[in]], #16    \n" /* load input, 16 int8 */
        "ldr    q9,     [%[w0]], #16    \n" /* load w0, 16 int8 */
        /* mul, lower 8 int8 * int8 = int16 */
        "smull  v18.8h, v8.8b, v9.8b    \n" /* mul in * w0, low, 8 int8 */
        "subs %w[cnt], %w[cnt], #1      \n" /* sub main loop count */
        /* mul, higher 8 int8 * int8 + int16 = int16 */
        "smlal2 v18.8h,v8.16b,v9.16b    \n" /* mul in * w0, high, 8 int8 */
        /* add int16 to int32 */
        "sadalp v0.4s, v18.8h           \n" /* pair acc, 8 int16 -> 4 int32 */
        "bne 1b                         \n" /* jump to main loop */
        /* pair add to final result */
        "2:                             \n" /* reduce to scale */
        "addv   s8, v0.4s               \n" /* reduction to out0 */
        /* write to output */
        "str s8, [%[out]]               \n" /* save result */
        : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop)
        : [out] "r"(ptr_out)
        : "cc", "memory", "v0", "v8", "v9", "v18");
    for (int i = 0; i < tail; ++i) {
      ptr_out[0] += ptr_in[i] * ptr_w0[i];
    }
    write_gemv_out(
        ptr_out, out_ptr, scale_ptr, bias_ptr, 1, flag_act, act, six, alpha);
  }
#else  //  __aarch64__
  int out_cnt = M >> 2;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 4;
    dtype* out_ptr = data_out + out_idx;
    const float* scale_ptr = scale + out_idx;
    int ptr_out[4] = {0, 0, 0, 0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = weights_ptr + (N * out_idx);
    const int8_t* ptr_w1 = ptr_w0 + N;
    const int8_t* ptr_w2 = ptr_w1 + N;
    const int8_t* ptr_w3 = ptr_w2 + N;
    int cnt_loop = cnt;
    auto bias_ptr = is_bias ? bias + out_idx : nullptr;
    asm volatile(
        "pld [%[in]]                    @ preload cache line, input\n"
        "pld [%[w0]]                    @ preload cache line, weights r0\n"
        "pld [%[w1]]                    @ preload cache line, weights r1\n"
        "pld [%[w2]]                    @ preload cache line, weights r2\n"
        "pld [%[w3]]                    @ preload cache line, weights r3\n"
        "vmov.u32 q0, #0                @ set q0 to 0\n"
        "vmov.u32 q1, #0                @ set q1 to 0\n"
        "vmov.u32 q2, #0                @ set q2 to 0\n"
        "vmov.u32 q3, #0                @ set q3 to 0\n"
        // "vld1.32 {d20-d21}, %[bias]     @ load bias data"
        "cmp %[cnt], #1                 @ check whether has main loop\n"
        "blt  2f                        @ jump to pair add\n"
        /* main loop */
        "1:                             @ main loop\n"
        "vld1.8 {d8-d9}, [%[in]]!       @ load input, q4\n"
        "vld1.8 {d12-d13}, [%[w0]]!     @ load weights r0, q6\n"
        "vld1.8 {d14-d15}, [%[w1]]!     @ load weights r1, q7\n"
        "vld1.8 {d16-d17}, [%[w2]]!     @ load weights r2, q8\n"
        "vld1.8 {d18-d19}, [%[w3]]!     @ load weights r3, q9\n"
        /* mul, int8 * int8 = int16 */
        "vmull.s8 q12, d8, d12          @ mul add\n"
        "vmull.s8 q13, d8, d14          @ mul add\n"
        "vmull.s8 q14, d8, d16          @ mul add\n"
        "vmull.s8 q15, d8, d18          @ mul add\n"
        /* mla, int8 * int8 + int16 = int16 */
        "vmlal.s8 q12, d9, d13          @ mul add\n"
        "vmlal.s8 q13, d9, d15          @ mul add\n"
        "vmlal.s8 q14, d9, d17          @ mul add\n"
        "vmlal.s8 q15, d9, d19          @ mul add\n"
        /* pacc, int16 + int32 = int32 */
        "vpadal.s16 q0, q12             @ pair acc\n"
        "vpadal.s16 q1, q13             @ pair acc\n"
        "vpadal.s16 q2, q14             @ pair acc\n"
        "vpadal.s16 q3, q15             @ pair acc\n"
        "subs %[cnt], #1                @ sub loop count \n"
        /* check loop end */
        "bne 1b                         @ jump to main loop\n"
        /* pair add to final result */
        "2:                             @ pair add \n"
        "vpadd.s32 d8, d0, d1           @ pair add, first step\n"
        "vpadd.s32 d9, d2, d3           @ pair add, first step\n"
        "vpadd.s32 d10, d4, d5          @ pair add, first step\n"
        "vpadd.s32 d11, d6, d7          @ pair add, first step\n"
        "vpadd.s32 d0, d8, d9           @ pair add, second step\n"
        "vpadd.s32 d1, d10, d11         @ pair add, second step\n"
        /* write output */
        "vst1.32 {d0-d1}, [%[out]]      @ save result\n"
        : [in] "+r"(ptr_in),
          [w0] "+r"(ptr_w0),
          [w1] "+r"(ptr_w1),
          [w2] "+r"(ptr_w2),
          [w3] "+r"(ptr_w3),
          [cnt] "+r"(cnt_loop)
        : [out] "r"(ptr_out)
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
          "q12",
          "q13",
          "q14",
          "q15");
    for (int i = 0; i < tail; ++i) {
      ptr_out[0] += ptr_in[i] * ptr_w0[i];
      ptr_out[1] += ptr_in[i] * ptr_w1[i];
      ptr_out[2] += ptr_in[i] * ptr_w2[i];
      ptr_out[3] += ptr_in[i] * ptr_w3[i];
    }
    write_gemv_out(
        ptr_out, out_ptr, scale_ptr, bias_ptr, 4, flag_act, act, six, alpha);
  }
//! deal with remains
#pragma omp parallel for
  for (int j = out_cnt * 4; j < M; j++) {
    dtype* out_ptr = data_out + j;
    const float* scale_ptr = scale + j;
    int ptr_out[1] = {0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    auto bias_ptr = is_bias ? bias + j : nullptr;
    asm volatile(
        "pld [%[in]]                        @ preload cache line, input\n"
        "pld [%[w0]]                        @ preload cache line, weights r0\n"
        "vmov.u32 q0, #0                    @ set q0 to 0\n"
        "cmp %[cnt], #1                     @ check whether has main loop\n"
        "blt  2f                            @ jump to tail\n"
        /* main loop */
        "1:                                 @ main loop\n"
        "vld1.8 {d24-d25}, [%[in]]!         @ load input, q12\n"
        "vld1.8 {d28-d29}, [%[w0]]!         @ load weights q14\n"
        /* mull int8 * int8 = int16*/
        "vmull.s8 q1, d24, d28              @ mul add\n"
        "vmlal.s8 q1, d25, d29              @ mul add\n"
        "subs %[cnt] , #1                   @ sub loop count \n"
        /* pacc int16 + int32 = int32*/
        "vpadal.s16 q0, q1                  @ pair acc\n"
        "bne 1b                             @ jump to main loop\n"
        /* pair add to final result */
        "2:                                 @ end processing\n"
        "vpadd.s32 d2, d0, d1               @ pair add, first step\n"
        "vpadd.s32 d0, d2, d2               @ pair add, final step\n"
        /* write output */
        "vst1.32 {d0[0]}, [%[out]]          @ save result\n"
        : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop)
        : [out] "r"(ptr_out)
        : "cc", "memory", "q0", "q1", "q12", "q13");
    for (int i = 0; i < tail; ++i) {
      ptr_out[0] += ptr_in[i] * ptr_w0[i];
    }
    write_gemv_out(
        ptr_out, out_ptr, scale_ptr, bias_ptr, 1, flag_act, act, six, alpha);
  }
#endif  //  __aarch64__
  return true;
}

#if defined(__aarch64__) && defined(WITH_ARM_DOTPROD)
template <typename dtype>
bool gemv_int8_sdot(const int8_t* A,
                    const int8_t* x,
                    dtype* y,
                    bool transA,
                    int M,
                    int N,
                    const float* scale,
                    bool is_bias,
                    const float* bias,
                    bool flag_act,
                    lite_api::ActivationType act,
                    float six,
                    float alpha) {
  if (transA) {
    LOG(ERROR) << "ERROR: sgemv, transA is not supported now";
    return false;
  }
  dtype* data_out = y;
  const int8_t* data_in = x;
  const int8_t* weights_ptr = A;
  int cnt = N >> 4;
  int tail = N & 15;
  int size_m = (M >> 3) << 3;
#pragma omp parallel for
  for (int j = 0; j < M - 7; j += 8) {
    dtype* out_ptr = data_out + j;
    const float* scale_ptr = scale + j;
    auto bias_ptr = is_bias ? bias + j : nullptr;
    int ptr_out[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = weights_ptr + (N * j);
    const int8_t* ptr_w1 = ptr_w0 + N;
    const int8_t* ptr_w2 = ptr_w1 + N;
    const int8_t* ptr_w3 = ptr_w2 + N;
    const int8_t* ptr_w4 = ptr_w3 + N;
    const int8_t* ptr_w5 = ptr_w4 + N;
    const int8_t* ptr_w6 = ptr_w5 + N;
    const int8_t* ptr_w7 = ptr_w6 + N;
    int cnt_loop = cnt;
    if (cnt > 0) {
      asm volatile(
          "prfm  pldl1keep, [%[in]]           \n" /* preload din */
          "prfm  pldl1keep, [%[w0]]   \n"         /* preload w0 */
          "prfm  pldl1keep, [%[w1]]   \n"         /* preload w1 */
          "prfm  pldl1keep, [%[w2]]   \n"         /* preload w2 */
          "prfm  pldl1keep, [%[w3]]   \n"         /* preload w3 */
          "prfm  pldl1keep, [%[w4]]   \n"         /* preload w4 */
          "prfm  pldl1keep, [%[w5]]   \n"         /* preload w5 */
          "prfm  pldl1keep, [%[w6]]   \n"         /* preload w6 */
          "prfm  pldl1keep, [%[w7]]   \n"         /* preload w7 */
          "movi   v0.4s,  #0          \n"         /* set out0 to 0 */
          "movi   v1.4s,  #0          \n"         /* set out1 to 0 */
          "movi   v2.4s,  #0          \n"         /* set out2 to 0 */
          "movi   v3.4s,  #0          \n"         /* set out3 to 0 */
          "movi   v4.4s,  #0          \n"         /* set out4 to 0 */
          "movi   v5.4s,  #0          \n"         /* set out5 to 0 */
          "movi   v6.4s,  #0          \n"         /* set out6 to 0 */
          "movi   v7.4s,  #0          \n"         /* set out7 to 0 */
          /* main loop */
          "1:                         \n" /* main loop */
          "ldr    q8,    [%[in]], #16 \n" /* load input, 16 int8 */
          "ldr    q9,   [%[w0]], #16 \n"  /* load w0, 16 int8 */
          "ldr    q10,   [%[w1]], #16 \n" /* load w0, 16 int8 */
          "ldr    q11,   [%[w2]], #16 \n" /* load w0, 16 int8 */
          "ldr    q12,   [%[w3]], #16 \n" /* load w0, 16 int8 */
          "ldr    q13,   [%[w4]], #16 \n" /* load w0, 16 int8 */
          "ldr    q14,   [%[w5]], #16 \n" /* load w0, 16 int8 */
          "ldr    q15,   [%[w6]], #16 \n" /* load w0, 16 int8 */
          "ldr    q16,   [%[w7]], #16 \n" /* load w0, 16 int8 */

          ".word 0x4e899500  // sdot   v0.4s, v8.16b, v9.16b \n"  /* out0, out1,
                                                                     out2, out3
                                                                     */
          ".word 0x4e8a9501  // sdot   v1.4s, v8.16b, v10.16b \n" /* out4, out5,
                                                                     out6, out7
                                                                     */
          ".word 0x4e8b9502  // sdot   v2.4s, v8.16b, v11.16b \n" /* out0, out1,
                                                                     out2, out3
                                                                     */
          ".word 0x4e8c9503  // sdot   v3.4s, v8.16b, v12.16b \n" /* out4, out5,
                                                                     out6, out7
                                                                     */
          "subs %w[cnt], %w[cnt], #1 \n"
          ".word 0x4e8d9504  // sdot   v4.4s, v8.16b, v13.16b \n" /* out0, out1,
                                                                     out2, out3
                                                                     */
          ".word 0x4e8e9505  // sdot   v5.4s, v8.16b, v14.16b \n" /* out4, out5,
                                                                     out6, out7
                                                                     */
          ".word 0x4e8f9506  // sdot   v6.4s, v8.16b, v15.16b \n" /* out0, out1,
                                                                     out2, out3
                                                                     */
          ".word 0x4e909507  // sdot   v7.4s, v8.16b, v16.16b \n" /* out4, out5,
                                                                     out6, out7
                                                                     */
          "bne 1b                      \n" /* jump to main loop */
          /* pair add to final result */
          "2:                          \n"  /* reduce to scale */
          "addp v10.4s , v0.4s , v1.4s  \n" /* pair add to 4 int32*/
          "addp v11.4s , v2.4s , v3.4s  \n" /* pair add to 4 int32*/
          "addp v12.4s , v4.4s , v5.4s  \n" /* pair add to 4 int32*/
          "addp v13.4s , v6.4s , v7.4s  \n" /* pair add to 4 int32*/

          "addp v0.4s , v10.4s , v11.4s  \n" /* pair add to 4 int32*/
          "addp v1.4s , v12.4s , v13.4s  \n" /* pair add to 4 int32*/
          /* write to output */
          "stp q0, q1, [%[out]]     \n" /* save result */
          : [in] "+r"(ptr_in),
            [w0] "+r"(ptr_w0),
            [w1] "+r"(ptr_w1),
            [w2] "+r"(ptr_w2),
            [w3] "+r"(ptr_w3),
            [w4] "+r"(ptr_w4),
            [w5] "+r"(ptr_w5),
            [w6] "+r"(ptr_w6),
            [w7] "+r"(ptr_w7),
            [cnt] "+r"(cnt_loop)
          : [out] "r"(ptr_out)
          : "cc",
            "memory",
            "v0",
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
            "v7",
            "v8",
            "v9",
            "v10",
            "v11",
            "v12",
            "v13",
            "v14",
            "v15",
            "v16",
            "v17",
            "v18");
    }
    for (int i = 0; i < tail; ++i) {
      ptr_out[0] += ptr_in[i] * ptr_w0[i];
      ptr_out[1] += ptr_in[i] * ptr_w1[i];
      ptr_out[2] += ptr_in[i] * ptr_w2[i];
      ptr_out[3] += ptr_in[i] * ptr_w3[i];
      ptr_out[4] += ptr_in[i] * ptr_w4[i];
      ptr_out[5] += ptr_in[i] * ptr_w5[i];
      ptr_out[6] += ptr_in[i] * ptr_w6[i];
      ptr_out[7] += ptr_in[i] * ptr_w7[i];
    }
    write_gemv_out(
        ptr_out, out_ptr, scale_ptr, bias_ptr, 8, flag_act, act, six, alpha);
  }
//! deal with remains
#pragma omp parallel for
  for (int j = size_m; j < M; j++) {
    // int *ptr_out = data_out + j;
    dtype* out_ptr = data_out + j;
    const float* scale_ptr = scale + j;
    int ptr_out[1] = {0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = weights_ptr + (N * j);
    int cnt_loop = cnt;
    auto bias_ptr = is_bias ? bias + j : nullptr;
    asm volatile(
        "prfm  pldl1keep, [%[in]]               \n" /* preload din */
        "prfm  pldl1keep, [%[w0]]       \n"         /* preload w0 */
        "cmp %w[cnt], #1                \n" /* check whether has main loop */
        "movi   v0.4s,  #0              \n" /* set out0 to 0 */
        /* check main loop */
        "blt  2f                        \n" /* jump to tail */
        /* main loop */
        "1:                             \n" /* main loop */
        "ldr    q8,     [%[in]], #16    \n" /* load input, 16 int8 */
        "ldr    q9,     [%[w0]], #16    \n" /* load w0, 16 int8 */
        "subs %w[cnt], %w[cnt], #1      \n" /* sub main loop count */
        /* mul, lower 8 int8 * int8 = int16 */
        ".word 0x4e899500  // sdot v0.4s, v8.16b, v9.16b \n"
        "bne 1b                         \n" /* jump to main loop */
        /* pair add to final result */
        "2:                             \n" /* reduce to scale */
        "addp   v1.4s, v0.4s, v0.4s     \n" /* reduction to out0 */
        "addp   v2.4s, v1.4s, v1.4s     \n" /* reduction to out0 */
        /* write to output */
        "str s2, [%[out]]               \n" /* save result */
        : [in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [cnt] "+r"(cnt_loop)
        : [out] "r"(ptr_out)
        : "cc", "memory", "v0", "v1", "v2", "v8", "v9", "v18");
    for (int i = 0; i < tail; ++i) {
      ptr_out[0] += ptr_in[i] * ptr_w0[i];
    }
    write_gemv_out(
        ptr_out, out_ptr, scale_ptr, bias_ptr, 1, flag_act, act, six, alpha);
  }
  return true;
}
#endif  // __aarch64__ && sdot

template <>
bool gemv_int8<float>(const int8_t* A,
                      const int8_t* x,
                      float* y,
                      bool transA,
                      int M,
                      int N,
                      const float* scale,
                      bool is_bias,
                      const float* bias,
                      bool flag_act,
                      lite_api::ActivationType act,
                      const ARMContext* ctx,
                      float six,
                      float alpha) {
#if defined(__aarch64__) && defined(WITH_ARM_DOTPROD)
  if (ctx->has_dot()) {
    return gemv_int8_sdot<float>(
        A, x, y, transA, M, N, scale, is_bias, bias, flag_act, act, six, alpha);
  } else {
    return gemv_int8_oth<float>(
        A, x, y, transA, M, N, scale, is_bias, bias, flag_act, act, six, alpha);
  }
#else
  return gemv_int8_oth<float>(
      A, x, y, transA, M, N, scale, is_bias, bias, flag_act, act, six, alpha);
#endif
}

template <>
bool gemv_int8<int8_t>(const int8_t* A,
                       const int8_t* x,
                       int8_t* y,
                       bool transA,
                       int M,
                       int N,
                       const float* scale,
                       bool is_bias,
                       const float* bias,
                       bool flag_act,
                       lite_api::ActivationType act,
                       const ARMContext* ctx,
                       float six,
                       float alpha) {
#if defined(__aarch64__) && defined(WITH_ARM_DOTPROD)
  if (ctx->has_dot()) {
    return gemv_int8_sdot<int8_t>(
        A, x, y, transA, M, N, scale, is_bias, bias, flag_act, act, six, alpha);
  } else {
    return gemv_int8_oth<int8_t>(
        A, x, y, transA, M, N, scale, is_bias, bias, flag_act, act, six, alpha);
  }
#else
  return gemv_int8_oth<int8_t>(
      A, x, y, transA, M, N, scale, is_bias, bias, flag_act, act, six, alpha);
#endif
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
