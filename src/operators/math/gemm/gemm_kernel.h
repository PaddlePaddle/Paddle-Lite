/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

#include <arm_neon.h>
#include "operators/math/math.h"

namespace paddle_mobile {
namespace operators {
namespace math {

#if __aarch64__
void sgemm_6x16(const float *lhs, const float *rhs, const int k, float *output,
                const int ldc) {
  int kc1 = k;
  int step = 4 * ldc;
  int step1 = 4 * 6;
  asm volatile(
      "dup      v6.4s,     wzr     \n\t"
      "dup      v7.4s,     wzr     \n\t"
      "dup      v8.4s,     wzr     \n\t"
      "dup      v9.4s,     wzr     \n\t"
      "dup      v10.4s,    wzr     \n\t"
      "dup      v11.4s,    wzr     \n\t"
      "dup      v12.4s,    wzr     \n\t"
      "dup      v13.4s,    wzr     \n\t"

      "dup      v14.4s,    wzr     \n\t"
      "dup      v15.4s,    wzr     \n\t"
      "dup      v16.4s,    wzr     \n\t"
      "dup      v17.4s,    wzr     \n\t"
      "dup      v18.4s,    wzr     \n\t"
      "dup      v19.4s,    wzr     \n\t"
      "dup      v20.4s,    wzr     \n\t"
      "dup      v21.4s,    wzr     \n\t"

      "dup      v22.4s,    wzr     \n\t"
      "dup      v23.4s,    wzr     \n\t"
      "dup      v24.4s,    wzr     \n\t"
      "dup      v25.4s,    wzr     \n\t"
      "dup      v26.4s,    wzr     \n\t"
      "dup      v27.4s,    wzr     \n\t"
      "dup      v28.4s,    wzr     \n\t"
      "dup      v29.4s,    wzr     \n\t"

      "subs     %[kc1], %[kc1], #1          \n\t"
      "blt      2f                          \n\t"
      "1:                                   \n\t"

      "prfm     pldl1keep,  [%[lhs],  #32]  \n\t"
      "prfm     pldl1keep,  [%[rhs],  #64]  \n\t"

      "ld1      {v0.4s, v1.4s},  [%[lhs]],    %[step1]      \n\t"
      "ld1      {v2.4s, v3.4s, v4.4s, v5.4s}, [%[rhs]], #64 \n\t"

      "fmla     v6.4s,    v2.4s,   v0.s[0]       \n\t"
      "fmla     v7.4s,    v3.4s,   v0.s[0]       \n\t"
      "fmla     v8.4s,    v4.4s,   v0.s[0]       \n\t"
      "fmla     v9.4s,    v5.4s,   v0.s[0]       \n\t"

      "fmla     v10.4s,   v2.4s,   v0.s[1]       \n\t"
      "fmla     v11.4s,   v3.4s,   v0.s[1]       \n\t"
      "fmla     v12.4s,   v4.4s,   v0.s[1]       \n\t"
      "fmla     v13.4s,   v5.4s,   v0.s[1]       \n\t"

      "fmla     v14.4s,   v2.4s,   v0.s[2]       \n\t"
      "fmla     v15.4s,   v3.4s,   v0.s[2]       \n\t"
      "fmla     v16.4s,   v4.4s,   v0.s[2]       \n\t"
      "fmla     v17.4s,   v5.4s,   v0.s[2]       \n\t"

      "fmla     v18.4s,   v2.4s,   v0.s[3]       \n\t"
      "fmla     v19.4s,   v3.4s,   v0.s[3]       \n\t"
      "fmla     v20.4s,   v4.4s,   v0.s[3]       \n\t"
      "fmla     v21.4s,   v5.4s,   v0.s[3]       \n\t"

      "fmla     v22.4s,   v2.4s,   v1.s[0]       \n\t"
      "fmla     v23.4s,   v3.4s,   v1.s[0]       \n\t"
      "fmla     v24.4s,   v4.4s,   v1.s[0]       \n\t"
      "fmla     v25.4s,   v5.4s,   v1.s[0]       \n\t"

      "fmla     v26.4s,   v2.4s,   v1.s[1]       \n\t"
      "fmla     v27.4s,   v3.4s,   v1.s[1]       \n\t"
      "fmla     v28.4s,   v4.4s,   v1.s[1]       \n\t"
      "fmla     v29.4s,   v5.4s,   v1.s[1]       \n\t"

      "subs       %[kc1], %[kc1], #1      \n\t"
      "bge        1b                      \n\t"
      "2:                                 \n\t"

      "st1      {v6.4s,  v7.4s,  v8.4s,  v9.4s},    [%[c]],   %[step]   \n\t"
      "st1      {v10.4s, v11.4s, v12.4s, v13.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v14.4s, v15.4s, v16.4s, v17.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v18.4s, v19.4s, v20.4s, v21.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v22.4s, v23.4s, v24.4s, v25.4s},   [%[c]],   %[step]   \n\t"
      "st1      {v26.4s, v27.4s, v28.4s, v29.4s},   [%[c]],   %[step]   \n\t"
      : [lhs] "+r"(lhs), [rhs] "+r"(rhs), [c] "+r"(output), [kc1] "+r"(kc1)
      : [step] "r"(step), [step1] "r"(step1)
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
        "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
        "v29");
}
#else
void sgemm_6x8(const float *lhs, const float *rhs, const int k, float *output,
               const int ldc) {
  int kc1 = k >> 3;   // k / 8
  int kc2 = k & 0x7;  // k % 8
  int step = sizeof(float) * ldc;
  asm volatile(
      "pld        [%[lhs]]            \n\t"
      "pld        [%[lhs],  #64]      \n\t"
      "pld        [%[rhs]]            \n\t"
      "pld        [%[rhs],  #64]      \n\t"

      "vmov.f32   q4,     #0.0          \n\t"
      "vmov.f32   q5,     #0.0          \n\t"
      "vmov.f32   q6,     #0.0          \n\t"
      "vmov.f32   q7,     #0.0          \n\t"
      "vmov.f32   q8,     #0.0          \n\t"
      "vmov.f32   q9,     #0.0          \n\t"
      "vmov.f32   q10,    #0.0          \n\t"
      "vmov.f32   q11,    #0.0          \n\t"
      "vmov.f32   q12,    #0.0          \n\t"
      "vmov.f32   q13,    #0.0          \n\t"
      "vmov.f32   q14,    #0.0          \n\t"
      "vmov.f32   q15,    #0.0          \n\t"

      "subs       %[kc1], %[kc1], #1    \n\t"
      "blt        2f                    \n\t"
      "1:                               \n\t"

      "pld        [%[lhs], #128]       \n\t"
      "pld        [%[rhs], #128]       \n\t"

      "vld1.32    {d0-d2},  [%[lhs]]!   \n\t"
      "vld1.32    {q2, q3}, [%[rhs]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "vld1.32    {d0-d2},  [%[lhs]]!   \n\t"
      "vld1.32    {q2, q3}, [%[rhs]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "pld        [%[lhs], #128]       \n\t"
      "pld        [%[rhs], #128]       \n\t"

      "vld1.32    {d0-d2},  [%[lhs]]!   \n\t"
      "vld1.32    {q2, q3}, [%[rhs]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "vld1.32    {d0-d2},  [%[lhs]]!   \n\t"
      "vld1.32    {q2, q3}, [%[rhs]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "pld        [%[lhs], #128]       \n\t"
      "pld        [%[rhs], #128]       \n\t"

      "vld1.32    {d0-d2},  [%[lhs]]!   \n\t"
      "vld1.32    {q2, q3}, [%[rhs]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "vld1.32    {d0-d2},  [%[lhs]]!   \n\t"
      "vld1.32    {q2, q3}, [%[rhs]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "pld        [%[lhs], #128]       \n\t"
      "pld        [%[rhs], #128]       \n\t"

      "vld1.32    {d0-d2},  [%[lhs]]!   \n\t"
      "vld1.32    {q2, q3}, [%[rhs]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "vld1.32    {d0-d2},  [%[lhs]]!   \n\t"
      "vld1.32    {q2, q3}, [%[rhs]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "subs       %[kc1], %[kc1], #1      \n\t"
      "bge        1b                      \n\t"
      "2:                                 \n\t"

      "subs       %[kc2], %[kc2], #1      \n\t"
      "blt        4f                      \n\t"
      "3:                                 \n\t"

      "vld1.32    {d0-d2},  [%[lhs]]!   \n\t"
      "vld1.32    {q2, q3}, [%[rhs]]!   \n\t"

      "vmla.f32   q4,   q2,   d0[0]       \n\t"
      "vmla.f32   q5,   q3,   d0[0]       \n\t"
      "vmla.f32   q6,   q2,   d0[1]       \n\t"
      "vmla.f32   q7,   q3,   d0[1]       \n\t"
      "vmla.f32   q8,   q2,   d1[0]       \n\t"
      "vmla.f32   q9,   q3,   d1[0]       \n\t"
      "vmla.f32   q10,  q2,   d1[1]       \n\t"
      "vmla.f32   q11,  q3,   d1[1]       \n\t"
      "vmla.f32   q12,  q2,   d2[0]       \n\t"
      "vmla.f32   q13,  q3,   d2[0]       \n\t"
      "vmla.f32   q14,  q2,   d2[1]       \n\t"
      "vmla.f32   q15,  q3,   d2[1]       \n\t"

      "subs       %[kc2], %[kc2], #1      \n\t"
      "bge        3b                      \n\t"
      "4:                                 \n\t"

      "mov        r5,     %[c]            \n\t"
      "mov        r6,     %[step]         \n\t"
      "vst1.32    {q4, q5},   [r5], r6    \n\t"
      "vst1.32    {q6, q7},   [r5], r6    \n\t"
      "vst1.32    {q8, q9},   [r5], r6    \n\t"
      "vst1.32    {q10, q11}, [r5], r6    \n\t"
      "vst1.32    {q12, q13}, [r5], r6    \n\t"
      "vst1.32    {q14, q15}, [r5]        \n\t"
      :
      : [lhs] "r"(lhs), [rhs] "r"(rhs), [c] "r"(output), [kc1] "r"(kc1),
        [kc2] "r"(kc2), [step] "r"(step)
      : "cc", "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
        "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}
#endif  // __aarch64__

void sgemv_notrans_mx1(const int M, const int N, const float alpha,
                       const float *A, const int lda, const float *B,
                       const float beta, float *C) {
  uint32_t mask[4] = {0, 1, 2, 3};
  int remain_n = N & 0x3;
  uint32x4_t vmask = vcltq_u32(vld1q_u32(mask), vdupq_n_u32(remain_n));
  float32x4_t _valpha = vdupq_n_f32(alpha);

  #pragma omp parallel for
  for (int m = 0; m < M - 3; m += 4) {
    const float *in0 = A + m * lda;
    const float *in1 = in0 + lda;
    const float *in2 = in1 + lda;
    const float *in3 = in2 + lda;
    float *output = C + m;

    float32x4_t _sum0, _sum1, _sum2, _sum3;
    _sum0 = vdupq_n_f32(0.f);
    _sum1 = vdupq_n_f32(0.f);
    _sum2 = vdupq_n_f32(0.f);
    _sum3 = vdupq_n_f32(0.f);
    int n = 0;
    for (; n < N - 3; n += 4) {
      float32x4_t _r0 = vld1q_f32(in0 + n);
      float32x4_t _r1 = vld1q_f32(in1 + n);
      float32x4_t _r2 = vld1q_f32(in2 + n);
      float32x4_t _r3 = vld1q_f32(in3 + n);
      float32x4_t _b = vld1q_f32(B + n);
      _sum0 = vmlaq_f32(_sum0, _r0, _b);
      _sum1 = vmlaq_f32(_sum1, _r1, _b);
      _sum2 = vmlaq_f32(_sum2, _r2, _b);
      _sum3 = vmlaq_f32(_sum3, _r3, _b);
    }
    if (n < N) {
      float32x4_t _r0 = vld1q_f32(in0 + n);
      float32x4_t _r1 = vld1q_f32(in1 + n);
      float32x4_t _r2 = vld1q_f32(in2 + n);
      float32x4_t _r3 = vld1q_f32(in3 + n);
      float32x4_t _b = vld1q_f32(B + n);
      _r0 = vandq_f32_u32(_r0, vmask);
      _r1 = vandq_f32_u32(_r1, vmask);
      _r2 = vandq_f32_u32(_r2, vmask);
      _r3 = vandq_f32_u32(_r3, vmask);
      _b = vandq_f32_u32(_b, vmask);
      _sum0 = vmlaq_f32(_sum0, _r0, _b);
      _sum1 = vmlaq_f32(_sum1, _r1, _b);
      _sum2 = vmlaq_f32(_sum2, _r2, _b);
      _sum3 = vmlaq_f32(_sum3, _r3, _b);
    }
    _sum0 = vpaddq_f32(_sum0, _sum1);
    _sum2 = vpaddq_f32(_sum2, _sum3);
    _sum0 = vpaddq_f32(_sum0, _sum2);
    _sum0 = vmulq_f32(_sum0, _valpha);
    if (beta != 0.f) {
      _sum2 = vmulq_n_f32(vld1q_f32(output), beta);
      _sum0 = vaddq_f32(_sum0, _sum2);
    }
    // restore
    vst1q_f32(output, _sum0);
  }
  // remain m
  for (int m = (M & 0xfffffffc); m < M; ++m) {
    const float *in0 = A + m * lda;
    float *output = C + m;
    float32x4_t _sum0 = vdupq_n_f32(0.f);

    int n = 0;
    for (; n < N - 3; n += 4) {
      float32x4_t _r0 = vld1q_f32(in0 + n);
      float32x4_t _b = vld1q_f32(B + n);
      _sum0 = vmlaq_f32(_sum0, _r0, _b);
    }
    if (n < N) {
      float32x4_t _r0 = vld1q_f32(in0 + n);
      float32x4_t _b = vld1q_f32(B + n);
      _r0 = vandq_f32_u32(_r0, vmask);
      _b = vandq_f32_u32(_b, vmask);
      _sum0 = vmlaq_f32(_sum0, _r0, _b);
    }
    _sum0 = vpaddq_f32(_sum0, _sum0);
    _sum0 = vmulq_f32(_sum0, _valpha);
    if (beta != 0.f) {
      float32x4_t _sum2 = vmulq_n_f32(vld1q_f32(output), beta);
      _sum0 = vpaddq_f32(_sum0, _sum2);
    }
    // restore
    *output = vgetq_lane_f32(_sum0, 0) + vgetq_lane_f32(_sum0, 1);
  }
}

void sgemv_trans_mx1_new(const int M, const int N, const float alpha,
                         const float *A, const int lda, const float *B,
                         const float beta, float *C) {
  // assign C with beta*C
  float32x4_t _valpha = vdupq_n_f32(alpha);
  if (beta == 0.f) {
    float32x4_t vzero = vdupq_n_f32(0.f);
    for (int m = 0; m < M - 3; m += 4) {
      vst1q_f32(C + m, vzero);
    }
    for (int m = (M & 0xfffffffc); m < M; ++m) {
      C[m] = 0.f;
    }
  } else {
    float32x4_t vbeta = vdupq_n_f32(beta);
    for (int m = 0; m < M - 3; m += 4) {
      float32x4_t _vc = vld1q_f32(C + m);
      _vc = vmulq_f32(_vc, vbeta);
      vst1q_f32(C + m, _vc);
    }
    for (int m = (M & 0xfffffffc); m < M; ++m) {
      C[m] *= beta;
    }
  }

  #pragma omp parallel for
  for (int m = 0; m < M - 3; m += 4) {
    // load A pointer
    register const float *ap = A + m;
    float32x4_t _sum = vdupq_n_f32(0.0f);
    float32x4_t _c00_10_20_30_vreg = vld1q_f32(C + m);
    int n = 0;
    for (; n < N - 3; n += 4) {
      // load a, b, c
      float32x4_t b_vreg = vld1q_f32(B + n);
      float32x4_t a00_10_20_30_vreg = vld1q_f32(ap + M * n);
      float32x4_t a01_11_21_31_vreg = vld1q_f32(ap + M * (n + 1));
      float32x4_t a02_12_22_32_vreg = vld1q_f32(ap + M * (n + 2));
      float32x4_t a03_13_23_33_vreg = vld1q_f32(ap + M * (n + 3));

      _sum = vmlaq_lane_f32(_sum, a00_10_20_30_vreg, vget_low_f32(b_vreg), 0);
      _sum = vmlaq_lane_f32(_sum, a01_11_21_31_vreg, vget_low_f32(b_vreg), 1);
      _sum = vmlaq_lane_f32(_sum, a02_12_22_32_vreg, vget_high_f32(b_vreg), 0);
      _sum = vmlaq_lane_f32(_sum, a03_13_23_33_vreg, vget_high_f32(b_vreg), 1);
    }

    // remain n, add to _sum
    for (; n < N; ++n) {
      float32x4_t a0n_1n_2n_3n_vreg = vld1q_f32(ap + M * n);
      float32x4_t bn_vreg = vdupq_n_f32(*(B + n));
      _sum = vmlaq_f32(_sum, a0n_1n_2n_3n_vreg, bn_vreg);
    }
    // _sum := _sum * valpha + _c
    _sum = vmlaq_f32(_c00_10_20_30_vreg, _sum, _valpha);
    // store
    vst1q_f32(C + m, _sum);
  }

  // remain m
  int remain_m = M & 3; // remain_m := M & (4-1)
  if (remain_m > 0) {
    const int remain_m_idx = M - remain_m;
    float32x4_t _c00_10_20_30_vreg = vld1q_f32(C + remain_m_idx);
    float32x4_t _sum = vdupq_n_f32(0.0f);
    register const float *ap = A + remain_m_idx;
    int n = 0;
    for (; n < N - 3; n += 4) {
      // load a, b
      float32x4_t a00_10_20_30_vreg = vld1q_f32(ap + M * (n));
      float32x4_t a01_11_21_31_vreg = vld1q_f32(ap + M * (n+1));
      float32x4_t a02_12_22_32_vreg = vld1q_f32(ap + M * (n+2));
      float32x4_t a03_13_23_33_vreg = vld1q_f32(ap + M * (n+3));
      float32x4_t b_vreg = vld1q_f32(B + n);

      _sum = vmlaq_lane_f32(_sum, a00_10_20_30_vreg, vget_low_f32(b_vreg), 0);
      _sum = vmlaq_lane_f32(_sum, a01_11_21_31_vreg, vget_low_f32(b_vreg), 1);
      _sum = vmlaq_lane_f32(_sum, a02_12_22_32_vreg, vget_high_f32(b_vreg), 0);
      _sum = vmlaq_lane_f32(_sum, a03_13_23_33_vreg, vget_high_f32(b_vreg), 1);
    }
    // remain n
    for (; n < N; ++n) {
      // load remain a, b
      float32x4_t a0n_1n_2n_3n_vreg = vld1q_f32(ap + M * n); 
      float32x4_t bn_vreg = vdupq_n_f32(*(B + n));
      _sum = vmlaq_f32(_sum, a0n_1n_2n_3n_vreg, bn_vreg);      
    }
    _sum = vmlaq_f32(_c00_10_20_30_vreg, _sum, _valpha);
    switch ( remain_m ) {
      case 3:
        vst1q_lane_f32(C + remain_m_idx + 2, _sum, 2);
      case 2:
        vst1_f32(C + remain_m_idx, vget_low_f32(_sum));
        break;
      case 1:
        vst1q_lane_f32(C + remain_m_idx, _sum, 0);
        break;
    }
  }
}

void sgemv_trans_mx1_old(const int M, const int N, const float alpha,
                         const float *A, const int lda, const float *B,
                         const float beta, float *C) {
  float32x4_t _valpha = vdupq_n_f32(alpha);
  if (beta == 0.f) {
    float32x4_t vzero = vdupq_n_f32(0.f);
    for (int m = 0; m < M - 3; m += 4) {
      vst1q_f32(C + m, vzero);
    }
    for (int m = (M & 0xfffffffc); m < M; ++m) {
      C[m] = 0.f;
    }
  } else {
    float32x4_t vbeta = vdupq_n_f32(beta);
    for (int m = 0; m < M - 3; m += 4) {
      float32x4_t _vc = vld1q_f32(C + m);
      _vc = vmulq_f32(_vc, vbeta);
      vst1q_f32(C + m, _vc);
    }
    for (int m = (M & 0xfffffffc); m < M; ++m) {
      C[m] *= beta;
    }
  }

  #pragma omp parallel for
  for (int n = 0; n < N - 3; n += 4) {
    const float *in0 = A + n * lda;
    const float *in1 = in0 + lda;
    const float *in2 = in1 + lda;
    const float *in3 = in2 + lda;
    float32x4_t _b = vld1q_f32(B + n);
    float32x4_t _sum0;
    int m = 0;
    for (; m < M - 3; m += 4) {
      float32x4_t _r0 = vld1q_f32(in0 + m);
      float32x4_t _r1 = vld1q_f32(in1 + m);
      float32x4_t _r2 = vld1q_f32(in2 + m);
      float32x4_t _r3 = vld1q_f32(in3 + m);
      float32x4_t _vc = vld1q_f32(C + m);

      _sum0 = vmulq_lane_f32(_r0, vget_low_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r1, vget_low_f32(_b), 1);
      _sum0 = vmlaq_lane_f32(_sum0, _r2, vget_high_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r3, vget_high_f32(_b), 1);
      _sum0 = vmulq_f32(_sum0, _valpha);
      _sum0 = vaddq_f32(_sum0, _vc);
      vst1q_f32(C + m, _sum0);
    }
    if (m < M) {
      float32x4_t _r0 = vld1q_f32(in0 + m);
      float32x4_t _r1 = vld1q_f32(in1 + m);
      float32x4_t _r2 = vld1q_f32(in2 + m);
      float32x4_t _r3 = vld1q_f32(in3 + m);
      float32x4_t _vc = vld1q_f32(C + m);

      _sum0 = vmulq_lane_f32(_r0, vget_low_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r1, vget_low_f32(_b), 1);
      _sum0 = vmlaq_lane_f32(_sum0, _r2, vget_high_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r3, vget_high_f32(_b), 1);
      _sum0 = vmulq_f32(_sum0, _valpha);
      _sum0 = vaddq_f32(_sum0, _vc);
      switch (M - m) {
        case 3:
          vst1q_lane_f32(C + m + 2, _sum0, 2);
        case 2:
          vst1_f32(C + m, vget_low_f32(_sum0));
          break;
        case 1:
          vst1q_lane_f32(C + m, _sum0, 0);
          break;
      }
    }
  }
  // remain n
  for (int n = (N & 0xfffffffc); n < N; ++n) {
    const float *in0 = A + n * lda;
    float32x4_t _b = vld1q_dup_f32(B + n);
    float32x4_t _sum0;
    int m = 0;
    for (; m < M - 3; m += 4) {
      float32x4_t _r0 = vld1q_f32(in0 + m);
      _sum0 = vld1q_f32(C + m);
      _r0 = vmulq_f32(_r0, _b);
      _r0 = vmulq_f32(_valpha, _r0);
      _sum0 = vaddq_f32(_sum0, _r0);
      vst1q_f32(C + m, _sum0);
    }
    for (; m < M; ++m) {
      C[m] += alpha * (in0[m] * B[n]);
    }
  }
}

void sgemv_mx1(const bool trans, const int M, const int N, const float alpha,
               const float *A, const int lda, const float *B, const float beta,
               float *C) {
  if (trans) {
    sgemv_trans_mx1_new(M, N, alpha, A, lda, B, beta, C);
  } else {
    sgemv_notrans_mx1(M, N, alpha, A, lda, B, beta, C);
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // __ARM_NEON__
