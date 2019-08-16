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
#include <string.h>
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
    float32x2_t _ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
    float32x2_t _sss2 = vpadd_f32(_ss, _ss);
    *output =
        vget_lane_f32(_sss2, 0) * vgetq_lane_f32(_valpha, 0) + beta * (*output);
  }
}

void sgemv_notrans_mx1_faster(const int M, const int N, const float alpha,
                              const float *A, const int lda, const float *B,
                              const float beta, float *C) {
#pragma omp parallel for
  for (int m = 0; m < M - 3; m += 4) {
    const float *a_ptr0 = A + m * lda;
    const float *a_ptr1 = a_ptr0 + lda;
    const float *a_ptr2 = a_ptr1 + lda;
    const float *a_ptr3 = a_ptr2 + lda;
    const float *b_ptr = B;
    float *c_ptr = C + m;
    float sum0 = 0.f;
    float sum1 = 0.f;
    float sum2 = 0.f;
    float sum3 = 0.f;
    int n = 0;

#if __ARM_NEON
    /* matrix_mul_float:
     * Calculate matrix A(4xN) * matrix B(Nx1) and store to a result array
     * sum_arr[4], a 4x8 * 8x1 will be calculated on each iteration.
     *
     * Variable: a_ptr0 = pointer to the first row of matrix A, row major order
     * Variable: a_ptr1 = pointer to the second row of matrix A, row major order
     * Variable: a_ptr2 = pointer to the third row of matrix A, row major order
     * Variable: a_ptr3 = pointer to the fourth row of matrix A, row major order
     * Variable: b_ptr  = pointer to the first col of matrix B, col major order
     * Variable: s_ptr  = pointer to the sum result array
     * Variable: loop   = the numbers of loops
     *
     * Register: Q(V)4-Q(V)11  = matrix A
     * Register: Q(V)0-Q(V)1   = matrix B
     * Register: Q(V)12-Q(V)15 = matrix C
     */

    float sum_arr[4] = {0.f};
    float *s_ptr = sum_arr;
    int loop = N / 8;

#if __aarch64__

    if (loop > 0) {
      asm volatile(
          // set v12-v15 to 0
          "movi   v12.4s,            #0             \n"
          "movi   v13.4s,            #0             \n"
          "movi   v14.4s,            #0             \n"
          "movi   v15.4s,            #0             \n"

          "0:                                       \n"
          // load A and B
          "ld1   {v0.4s, v1.4s},   [%[b_ptr]] , #32 \n"
          "ld1   {v4.4s, v5.4s},   [%[a_ptr0]], #32 \n"
          "ld1   {v6.4s, v7.4s},   [%[a_ptr1]], #32 \n"
          "ld1   {v8.4s, v9.4s},   [%[a_ptr2]], #32 \n"
          "ld1   {v10.4s, v11.4s}, [%[a_ptr3]], #32 \n"

          "fmla   v12.4s, v4.4s,  v0.4s             \n"  // s0=A(r0c0-r0c3)*B(r0-r3)
          "fmla   v13.4s, v6.4s,  v0.4s             \n"  // s1=A(r1c0-r1c3)*B(r0-r3)
          "fmla   v14.4s, v8.4s,  v0.4s             \n"  // s2=A(r2c0-r2c3)*B(r0-r3)
          "fmla   v15.4s, v10.4s, v0.4s             \n"  // s3=A(r3c0-r3c3)*B(r0-r3)

          "fmla   v12.4s, v5.4s,  v1.4s             \n"  // s0=A(r0c4-r0c7)*B(r4-r7)
          "fmla   v13.4s, v7.4s,  v1.4s             \n"  // s1=A(r1c4-r1c7)*B(r4-r7)
          "fmla   v14.4s, v9.4s,  v1.4s             \n"  // s2=A(r2c4-r2c7)*B(r4-r7)
          "fmla   v15.4s, v11.4s, v1.4s             \n"  // s3=A(r3c4-r3c7)*B(r4-r7)

          // cycle
          "subs   %[loop], %[loop], #1              \n"
          "bne    0b                                \n"

          // add and store
          "faddp   v4.4s, v12.4s, v13.4s            \n"
          "faddp   v5.4s, v14.4s, v15.4s            \n"
          "faddp   v6.4s, v4.4s, v5.4s              \n"
          "st1    {v6.4s}, [%[s_ptr]]               \n"

          : [loop] "+r"(loop), [a_ptr0] "+r"(a_ptr0), [a_ptr1] "+r"(a_ptr1),
            [a_ptr2] "+r"(a_ptr2), [a_ptr3] "+r"(a_ptr3), [b_ptr] "+r"(b_ptr)
          : [s_ptr] "r"(s_ptr)
          : "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
            "v13", "v14", "v15", "cc", "memory");
    }
#else   // __aarch64__

    if (loop > 0) {
      asm volatile(

          // set Q12-Q15 to 0
          "vmov.i32    q12,       #0           \n"
          "vmov.i32    q13,       #0           \n"
          "vmov.i32    q14,       #0           \n"
          "vmov.i32    q15,       #0           \n"

          "0:                                  \n"
          // load A and B
          "vld1.f32    {d0-d3},   [%[b_ptr]]!  \n"
          "vld1.f32    {d8-d11},  [%[a_ptr0]]! \n"
          "vld1.f32    {d12-d15}, [%[a_ptr1]]! \n"
          "vld1.f32    {d16-d19}, [%[a_ptr2]]! \n"
          "vld1.f32    {d20-d23}, [%[a_ptr3]]! \n"

          "vmla.f32    q12, q4,   q0           \n"  // s0=A(r0c0-r0c3)*B(r0-r3)
          "vmla.f32    q13, q6,   q0           \n"  // s1=A(r1c0-r1c3)*B(r0-r3)
          "vmla.f32    q14, q8,   q0           \n"  // s2=A(r2c0-r2c3)*B(r0-r3)
          "vmla.f32    q15, q10,  q0           \n"  // s3=A(r3c0-r3c3)*B(r0-r3)

          "vmla.f32    q12, q5,   q1           \n"  // s0=A(r0c4-r0c7)*B(r4-r7)
          "vmla.f32    q13, q7,   q1           \n"  // s1=A(r1c4-r1c7)*B(r4-r7)
          "vmla.f32    q14, q9,   q1           \n"  // s2=A(r2c4-r2c7)*B(r4-r7)
          "vmla.f32    q15, q11,  q1           \n"  // s3=A(r3c4-r3c7)*B(r4-r7)

          // cycle
          "subs        %[loop],   #1           \n"
          "bne         0b                      \n"
          // add and store
          "vpadd.f32   d8, d24,   d25          \n"
          "vpadd.f32   d9, d26,   d27          \n"
          "vpadd.f32   d10, d28,  d29          \n"
          "vpadd.f32   d11, d30,  d31          \n"

          "vpadd.f32   d12, d8,   d9           \n"
          "vpadd.f32   d13, d10,  d11          \n"
          "vst1.32     {d12-d13}, [%[s_ptr]]   \n"

          : [loop] "+r"(loop), [a_ptr0] "+r"(a_ptr0), [a_ptr1] "+r"(a_ptr1),
            [a_ptr2] "+r"(a_ptr2), [a_ptr3] "+r"(a_ptr3), [b_ptr] "+r"(b_ptr)
          : [s_ptr] "r"(s_ptr)
          : "q0", "q1", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12",
            "q13", "q14", "q15", "cc", "memory");
    }
#endif  // __aarch64__
    sum0 += s_ptr[0];
    sum1 += s_ptr[1];
    sum2 += s_ptr[2];
    sum3 += s_ptr[3];
    n = N - (N & 0x07);
#endif  // __ARM_NEON

    for (; n < N - 7; n += 8) {
      sum0 += a_ptr0[0] * b_ptr[0];
      sum1 += a_ptr1[0] * b_ptr[0];
      sum2 += a_ptr2[0] * b_ptr[0];
      sum3 += a_ptr3[0] * b_ptr[0];

      sum0 += a_ptr0[1] * b_ptr[1];
      sum1 += a_ptr1[1] * b_ptr[1];
      sum2 += a_ptr2[1] * b_ptr[1];
      sum3 += a_ptr3[1] * b_ptr[1];

      sum0 += a_ptr0[2] * b_ptr[2];
      sum1 += a_ptr1[2] * b_ptr[2];
      sum2 += a_ptr2[2] * b_ptr[2];
      sum3 += a_ptr3[2] * b_ptr[2];

      sum0 += a_ptr0[3] * b_ptr[3];
      sum1 += a_ptr1[3] * b_ptr[3];
      sum2 += a_ptr2[3] * b_ptr[3];
      sum3 += a_ptr3[3] * b_ptr[3];

      sum0 += a_ptr0[4] * b_ptr[4];
      sum1 += a_ptr1[4] * b_ptr[4];
      sum2 += a_ptr2[4] * b_ptr[4];
      sum3 += a_ptr3[4] * b_ptr[4];

      sum0 += a_ptr0[5] * b_ptr[5];
      sum1 += a_ptr1[5] * b_ptr[5];
      sum2 += a_ptr2[5] * b_ptr[5];
      sum3 += a_ptr3[5] * b_ptr[5];

      sum0 += a_ptr0[6] * b_ptr[6];
      sum1 += a_ptr1[6] * b_ptr[6];
      sum2 += a_ptr2[6] * b_ptr[6];
      sum3 += a_ptr3[6] * b_ptr[6];

      sum0 += a_ptr0[7] * b_ptr[7];
      sum1 += a_ptr1[7] * b_ptr[7];
      sum2 += a_ptr2[7] * b_ptr[7];
      sum3 += a_ptr3[7] * b_ptr[7];

      a_ptr0 += 8;
      a_ptr1 += 8;
      a_ptr2 += 8;
      a_ptr3 += 8;
      b_ptr += 8;
    }

    for (; n < N; ++n) {
      sum0 += a_ptr0[0] * b_ptr[0];
      sum1 += a_ptr1[0] * b_ptr[0];
      sum2 += a_ptr2[0] * b_ptr[0];
      sum3 += a_ptr3[0] * b_ptr[0];

      a_ptr0 += 1;
      a_ptr1 += 1;
      a_ptr2 += 1;
      a_ptr3 += 1;
      b_ptr += 1;
    }
    c_ptr[0] = alpha * sum0 + beta * c_ptr[0];
    c_ptr[1] = alpha * sum1 + beta * c_ptr[1];
    c_ptr[2] = alpha * sum2 + beta * c_ptr[2];
    c_ptr[3] = alpha * sum3 + beta * c_ptr[3];
  }

  int m_tail_start = M - (M & 0x03);
  for (int m = m_tail_start; m < M; ++m) {
    const float *a_ptr = A + m * lda;
    const float *b_ptr = B;
    float *c_ptr = C + m;
    float sum = 0.f;
    for (int n = 0; n < N; n++) {
      sum += a_ptr[0] * b_ptr[0];
      a_ptr += 1;
      b_ptr += 1;
    }
    c_ptr[0] = alpha * sum + beta * c_ptr[0];
  }
}

void sgemv_trans_mx1(const int M, const int N, const float alpha,
                     const float *A, const int lda, const float *B,
                     const float beta, float *C) {
// create buff_c to store temp computation result for each threading
#ifdef _OPENMP
  int threads_num = omp_get_max_threads();
#else
  int threads_num = 1;
#endif  // _OPENMP
  float *buf_c = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * threads_num * M));
  memset(buf_c, 0, threads_num * M * sizeof(float));

  #pragma omp parallel for
  for (int n = 0; n < N - 3; n += 4) {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif  // _OPENMP
    float *thread_buf_c = buf_c + tid * M;
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
      float32x4_t _vbuff_c = vld1q_f32(thread_buf_c + m);

      _sum0 = vmulq_lane_f32(_r0, vget_low_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r1, vget_low_f32(_b), 1);
      _sum0 = vmlaq_lane_f32(_sum0, _r2, vget_high_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r3, vget_high_f32(_b), 1);
      _sum0 = vaddq_f32(_sum0, _vbuff_c);

      vst1q_f32(thread_buf_c + m, _sum0);
    }
    if (m < M) {
      float32x4_t _sum0 = vdupq_n_f32(0.0f);
      float32x4_t _r0 = vld1q_f32(in0 + m);
      float32x4_t _r1 = vld1q_f32(in1 + m);
      float32x4_t _r2 = vld1q_f32(in2 + m);
      float32x4_t _r3 = vld1q_f32(in3 + m);
      float32x4_t _vbuff_c = vld1q_f32(thread_buf_c + m);

      _sum0 = vmulq_lane_f32(_r0, vget_low_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r1, vget_low_f32(_b), 1);
      _sum0 = vmlaq_lane_f32(_sum0, _r2, vget_high_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r3, vget_high_f32(_b), 1);
      _sum0 = vaddq_f32(_sum0, _vbuff_c);
      switch (M - m) {
        case 3:
          vst1q_lane_f32(thread_buf_c + m + 2, _sum0, 2);
        case 2:
          vst1_f32(thread_buf_c + m, vget_low_f32(_sum0));
          break;
        case 1:
          vst1q_lane_f32(thread_buf_c + m, _sum0, 0);
          break;
      }
    }
  }

  // remain n
  #pragma omp parallel for
  for (int n = (N & 0xfffffffc); n < N; ++n) {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif  // _OPENMP
    float *thread_buf_c = buf_c + tid * M;
    const float *in0 = A + n * lda;
    float32x4_t _b = vld1q_dup_f32(B + n);
    float32x4_t _sum0;
    int m = 0;
    for (; m < M - 3; m += 4) {
      float32x4_t _r0 = vld1q_f32(in0 + m);
      float32x4_t _vbuff_c = vld1q_f32(thread_buf_c + m);
      _sum0 = vmulq_f32(_r0, _b);
      _sum0 = vaddq_f32(_sum0, _vbuff_c);
      vst1q_f32(thread_buf_c + m, _sum0);
    }
    for (; m < M; ++m) {
      thread_buf_c[m] += in0[m] * B[n];
    }
  }

  // reduction operate for buf_c, sum to C and do left operations
  // y := alpha * A' * X + beta * y
  // reduction operate: sum multi-threadings result for over-all: A' * X
  float32x4_t _valpha = vdupq_n_f32(alpha);
  if (beta == 0.f) {
    #pragma omp parallel for
    for (int m = 0; m < M - 3; m += 4) {
      float32x4_t _sum0 = vld1q_f32(buf_c + m);
      for (int tid = 1; tid < threads_num; ++tid) {
        _sum0 += vld1q_f32(buf_c + tid * M + m);
      }
      vst1q_f32(C + m, _sum0 * _valpha);
    }

    for (int m = (M & 0xfffffffc); m < M; ++m) {
      float _sum0 = *(buf_c + m);
      for (int tid = 1; tid < threads_num; ++tid) {
        _sum0 += *(buf_c + tid * M + m);
      }
      C[m] = _sum0 * alpha;
    }
  } else {  // beta != 0.f
    float32x4_t _vbeta = vdupq_n_f32(beta);
    #pragma omp parallel for
    for (int m = 0; m < M - 3; m += 4) {
      float32x4_t _sum0 = vld1q_f32(buf_c + m);
      for (int tid = 1; tid < threads_num; ++tid) {
        _sum0 += vld1q_f32(buf_c + tid * M + m);
      }
      float32x4_t _vc = vld1q_f32(C + m);
      vst1q_f32(C + m, _sum0 * _valpha + _vbeta * _vc);
    }

    for (int m = (M & 0xfffffffc); m < M; ++m) {
      float _sum0 = *(buf_c + m);
      for (int tid = 1; tid < threads_num; ++tid) {
        _sum0 += *(buf_c + tid * M + m);
      }
      C[m] = _sum0 * alpha + beta * C[m];
    }
  }

  // free buff_c
  paddle_mobile::memory::Free(buf_c);
}

void sgemv_mx1(const bool trans, const int M, const int N, const float alpha,
               const float *A, const int lda, const float *B, const float beta,
               float *C) {
  if (trans) {
    sgemv_trans_mx1(M, N, alpha, A, lda, B, beta, C);
  } else {
    //    sgemv_notrans_mx1(M, N, alpha, A, lda, B, beta, C);
    sgemv_notrans_mx1_faster(M, N, alpha, A, lda, B, beta, C);
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // __ARM_NEON__
