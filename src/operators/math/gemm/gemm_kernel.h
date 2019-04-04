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
  std::cout << "sgemv_notrans_mx1" << std::endl;
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

void sgemv_trans_mx1(const int M, const int N, const float alpha,
                     const float *A, const int lda, const float *B,
                     const float beta, float *C) {
  std::cout << "sgemv_trans_mx1" << std::endl;
  std::cout << "M=" << M << " N=" << N << " alpha=" << alpha << " lda=" << lda << " beta=" << beta << std::endl;

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

void sgemv_trans_mx1_v2(const int M, const int N, const float alpha,
                        const float *A, const int lda, const float *B,
                        const float beta, float *C) {
  std::cout << "sgemv_trans_mx1_v2" << std::endl;
  std::cout << "M=" << M << " N=" << N << " alpha=" << alpha << " lda=" << lda << " beta=" << beta << std::endl;

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

  for (int m = 0; m < M - 3; m += 4)
  {
    // load A4x1 pointers
    const float *a00 = A + m * lda;
    const float *a10 = a00 + lda;
    const float *a20 = a10 + lda;
    const float *a30 = a20 + lda;
    // init sum0
    float32x4_t sum0_1x4 = vdupq_n_f32(0.0f);
    float32x4_t sum1_1x4 = vdupq_n_f32(0.0f);
    float32x4_t sum2_1x4 = vdupq_n_f32(0.0f);
    float32x4_t sum3_1x4 = vdupq_n_f32(0.0f);
    
    int n = 0;
    for (; n < N - 3; n += 4)
    {
      // load A4x4
      float32x4_t a_00_a_01_a_02_a_03_vreg = vld1q_f32(a00 + n);
      float32x4_t a_10_a_11_a_12_a_13_vreg = vld1q_f32(a10 + n);
      float32x4_t a_20_a_21_a_22_a_23_vreg = vld1q_f32(a20 + n);
      float32x4_t a_30_a_31_a_32_a_33_vreg = vld1q_f32(a30 + n);

      // load B4x1
      float32x4_t b_00_b_10_b_20_b_30_vreg = vld1q_f32(B + n);

      // sum := alpha * (a_1x4)x4 * b_4x1
      a_00_a_01_a_02_a_03_vreg = vmulq_f32(a_00_a_01_a_02_a_03_vreg, _valpha);
      a_10_a_11_a_12_a_13_vreg = vmulq_f32(a_10_a_11_a_12_a_13_vreg, _valpha);
      a_20_a_21_a_22_a_23_vreg = vmulq_f32(a_20_a_21_a_22_a_23_vreg, _valpha);
      a_30_a_31_a_32_a_33_vreg = vmulq_f32(a_30_a_31_a_32_a_33_vreg, _valpha);

      sum0_1x4 = vmlaq_f32(sum0_1x4, a_00_a_01_a_02_a_03_vreg, b_00_b_10_b_20_b_30_vreg);
      sum1_1x4 = vmlaq_f32(sum1_1x4, a_10_a_11_a_12_a_13_vreg, b_00_b_10_b_20_b_30_vreg);
      sum2_1x4 = vmlaq_f32(sum2_1x4, a_20_a_21_a_22_a_23_vreg, b_00_b_10_b_20_b_30_vreg);
      sum3_1x4 = vmlaq_f32(sum3_1x4, a_30_a_31_a_32_a_33_vreg, b_00_b_10_b_20_b_30_vreg);
    }
    // save C4x1
    vst1q_f32(C+m,   vaddv_f32( vmulq_f32(sum0_1x4, _valpha) ) );
    vst1q_f32(C+m+1, vaddv_f32( vmulq_f32(sum1_1x4, _valpha) ) );
    vst1q_f32(C+m+2, vaddv_f32( vmulq_f32(sum2_1x4, _valpha) ) );
    vst1q_f32(C+m+3, vaddv_f32( vmulq_f32(sum3_1x4, _valpha) ) );

    switch ( N - n)
    {
      case 3:
      {
        register float c_00 = *(C+n);
        register float c_10 = *(C+n+1);
        register float c_20 = *(C+n+2);
        register float c_30 = *(C+n+3);

        c_00 += alpha * ( *(a00+n) * *(B+n) + *(a00+n+1) * *(B+n+1) + *(a00+n+2) * *(B+n+2) );
        c_10 += alpha * ( *(a10+n) * *(B+n) + *(a10+n+1) * *(B+n+1) + *(a10+n+2) * *(B+n+2) );
        c_20 += alpha * ( *(a20+n) * *(B+n) + *(a20+n+1) * *(B+n+1) + *(a20+n+2) * *(B+n+2) );
        c_30 += alpha * ( *(a30+n) * *(B+n) + *(a30+n+1) * *(B+n+1) + *(a30+n+2) * *(B+n+2) );
      }
        break;
      case 2:
      {
        float32x2_t _valphax2 = vdup_n_f32(alpha);
        // load a, b,  init result
        float32x2_t a_00_a_01_vreg = vld1_f32(a00 + n);
        float32x2_t a_10_a_11_vreg = vld1_f32(a10 + n);
        float32x2_t a_20_a_21_vreg = vld1_f32(a20 + n);
        float32x2_t a_30_a_31_vreg = vld1_f32(a30 + n);
        float32x2_t b_00_b_10_vreg = vld1_f32(B + n);
        float32x2_t res0 = vdup_n_f32(0.0f);
        float32x2_t res1 = vdup_n_f32(0.0f);
        float32x2_t res2 = vdup_n_f32(0.0f);
        float32x2_t res3 = vdup_n_f32(0.0f);

        // res := alpha * a * b
        a_00_a_01_vreg = vmul_f32(a_00_a_01_vreg, _valphax2);
        a_10_a_11_vreg = vmul_f32(a_10_a_11_vreg, _valphax2);
        a_20_a_21_vreg = vmul_f32(a_20_a_21_vreg, _valphax2);
        a_30_a_31_vreg = vmul_f32(a_30_a_31_vreg, _valphax2);

        res0 = vmla_f32(res0, a_00_a_01_vreg, b_00_b_10_vreg);
        res1 = vmla_f32(res1, a_10_a_11_vreg, b_00_b_10_vreg);
        res2 = vmla_f32(res2, a_20_a_21_vreg, b_00_b_10_vreg);
        res3 = vmla_f32(res3, a_30_a_31_vreg, b_00_b_10_vreg);

        float32x4_t c_00_c_10_c_20_c_30_vreg = (float32x4_t){vaddv_f32(res0),
                                                             vaddv_f32(res1),
                                                             vaddv_f32(res2),
                                                             vaddv_f32(res3)};
        vst1q_f32(C+n, c_00_c_10_c_20_c_30_vreg);
      }
        break;
      case 1:
      {
        register float a_00 = *(a00 + n);
        register float a_10 = *(a10 + n);
        register float a_20 = *(a20 + n);
        register float a_30 = *(a30 + n);
        register float b_00 = *(B + n);
        *(C+n)   = a_00 * b_00 * alpha;
        *(C+n+1) = a_10 * b_00 * alpha;
        *(C+n+2) = a_20 * b_00 * alpha;
        *(C+n+3) = a_30 * b_00 * alpha;
      }
        break;   
    }
  }

  // remain m
  for (int m = (M & 0xfffffffc); m < M; ++m) 
  {
    const float *a = A + m * lda;
    int n = 0;
    float32x4_t c_4x1_vreg = vdupq_n_f32(0.f);
    for (; n < N - 3; n += 4)
    {
      // load a, b
      float32x4_t a_1x4_vreg = vld1q_f32(a + n);
      float32x4_t b_4x1_vreg = vld1q_f32(B + n);
      // c := a*b*alpha
      c_4x1_vreg = vmulq_f32(a_1x4_vreg, b_4x1_vreg);
      c_4x1_vreg = vmulq_f32(c_4x1_vreg, _valpha);
    }
    vst1q_f32(C+n, c_4x1_vreg);
    for (; n < N; ++n)
    {
      C[n] += alpha * a[n] * B[n];
    }
  }
}

void sgemv_mx1(const bool trans, const int M, const int N, const float alpha,
               const float *A, const int lda, const float *B, const float beta,
               float *C) {
  if (trans) {
    sgemv_trans_mx1(M, N, alpha, A, lda, B, beta, C);
  } else {
    sgemv_notrans_mx1(M, N, alpha, A, lda, B, beta, C);
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // __ARM_NEON__
