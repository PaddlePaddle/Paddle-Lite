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

#include <string.h>
#include "common/log.h"
#include "operators/math/gemm.h"
#if __ARM_NEON
#include <arm_neon.h>
#include <iostream>

#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {
void Gemm::AddDot4x8(int32_t k, const int8_t *a, const int8_t *b, int32_t *c,
                     int32_t ldc) {
#if __ARM_NEON
#if __aarch64__
// AddDot4x8 used only for aarch32
#else
  const int8_t *a_ptr, *b_ptr;
  a_ptr = a;
  b_ptr = b;
  int32_t kc1 = k >> 3;
  int32_t kc2 = k & 7;
  int32_t kc3 = kc2 >> 2;
  int32_t kc4 = kc2 & 3;
  int32_t kc5 = kc4 >> 1;
  int32_t kc6 = kc4 & 1;
  int32_t step = sizeof(int32_t) * ldc;
  asm volatile(
      // q8-q15: save 32 results
      "pld          [%[a_ptr]]                     \n\t"
      "pld          [%[b_ptr]]                     \n\t"
      "pld          [%[b_ptr], #64]                \n\t"
      "vmov.s32     q8,         #0                 \n\t"
      "vmov.s32     q9,         q8                 \n\t"
      "vmov.s32     q10,        q8                 \n\t"
      "vmov.s32     q11,        q8                 \n\t"
      "vmov.s32     q12,        q8                 \n\t"
      "vmov.s32     q13,        q8                 \n\t"
      "vmov.s32     q14,        q8                 \n\t"
      "vmov.s32     q15,        q8                 \n\t"
      "subs         %[kc1],     %[kc1],       #1   \n\t"
      "blt          1f                             \n\t"
      "0:                                          \n\t"
      "pld          [%[a_ptr], #64]                \n\t"
      "pld          [%[b_ptr], #128]               \n\t"
      "vld1.s8      {d0-d3},    [%[a_ptr]]!        \n\t"  // load A 8 cols
      "vld1.s8      {d8-d11},   [%[b_ptr]]!        \n\t"  // load B first 4 rows
      "vmovl.s8     q2,         d0                 \n\t"  // process B first
                                                          // rows
      "vmovl.s8     q3,         d8                 \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "vmovl.s8     q3,         d9                 \n\t"
      "vmlal.s16    q8,         d6,            d5[0]\n\t"
      "vmlal.s16    q9,         d7,            d5[0]\n\t"
      "vmlal.s16    q10,        d6,            d5[1]\n\t"
      "vmlal.s16    q11,        d7,            d5[1]\n\t"
      "vmlal.s16    q12,        d6,            d5[2]\n\t"
      "vmlal.s16    q13,        d7,            d5[2]\n\t"
      "vmlal.s16    q14,        d6,            d5[3]\n\t"
      "vmlal.s16    q15,        d7,            d5[3]\n\t"
      "vld1.s8      {d12-d15},  [%[b_ptr]]!        \n\t"  // load B second 4
                                                          // rows
      "vmovl.s8     q2,         d1                 \n\t"
      "vmovl.s8     q3,         d10                 \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "vmovl.s8     q3,         d11                 \n\t"
      "vmlal.s16    q8,         d6,            d5[0]\n\t"
      "vmlal.s16    q9,         d7,            d5[0]\n\t"
      "vmlal.s16    q10,        d6,            d5[1]\n\t"
      "vmlal.s16    q11,        d7,            d5[1]\n\t"
      "vmlal.s16    q12,        d6,            d5[2]\n\t"
      "vmlal.s16    q13,        d7,            d5[2]\n\t"
      "vmlal.s16    q14,        d6,            d5[3]\n\t"
      "vmlal.s16    q15,        d7,            d5[3]\n\t"
      "vmovl.s8     q2,         d2                 \n\t"  // process B second 4
                                                          // rows
      "vmovl.s8     q3,         d12                 \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "vmovl.s8     q3,         d13                 \n\t"
      "vmlal.s16    q8,         d6,            d5[0]\n\t"
      "vmlal.s16    q9,         d7,            d5[0]\n\t"
      "vmlal.s16    q10,        d6,            d5[1]\n\t"
      "vmlal.s16    q11,        d7,            d5[1]\n\t"
      "vmlal.s16    q12,        d6,            d5[2]\n\t"
      "vmlal.s16    q13,        d7,            d5[2]\n\t"
      "vmlal.s16    q14,        d6,            d5[3]\n\t"
      "vmlal.s16    q15,        d7,            d5[3]\n\t"
      "vmovl.s8     q2,         d3                 \n\t"
      "vmovl.s8     q3,         d14                 \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "vmovl.s8     q3,         d15                 \n\t"
      "vmlal.s16    q8,         d6,            d5[0]\n\t"
      "vmlal.s16    q9,         d7,            d5[0]\n\t"
      "vmlal.s16    q10,        d6,            d5[1]\n\t"
      "vmlal.s16    q11,        d7,            d5[1]\n\t"
      "vmlal.s16    q12,        d6,            d5[2]\n\t"
      "vmlal.s16    q13,        d7,            d5[2]\n\t"
      "vmlal.s16    q14,        d6,            d5[3]\n\t"
      "vmlal.s16    q15,        d7,            d5[3]\n\t"

      "subs         %[kc1],     %[kc1],        #1  \n\t"
      "bge          0b                             \n\t"
      "1:                                          \n\t"  // last 4 rows
      "subs         %[kc3],     %[kc3],        #1  \n\t"
      "blt          2f                             \n\t"
      "vld1.s8      {d0-d1},    [%[a_ptr]]!        \n\t"  // load A 4 cols
      "vld1.s8      {d8-d11},   [%[b_ptr]]!        \n\t"  // load B 4 rows
      "vmovl.s8     q2,         d0                 \n\t"
      "vmovl.s8     q3,         d8                 \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "vmovl.s8     q3,         d9                 \n\t"
      "vmlal.s16    q8,         d6,            d5[0]\n\t"
      "vmlal.s16    q9,         d7,            d5[0]\n\t"
      "vmlal.s16    q10,        d6,            d5[1]\n\t"
      "vmlal.s16    q11,        d7,            d5[1]\n\t"
      "vmlal.s16    q12,        d6,            d5[2]\n\t"
      "vmlal.s16    q13,        d7,            d5[2]\n\t"
      "vmlal.s16    q14,        d6,            d5[3]\n\t"
      "vmlal.s16    q15,        d7,            d5[3]\n\t"
      "vmovl.s8     q2,         d1                 \n\t"
      "vmovl.s8     q3,         d10                 \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "vmovl.s8     q3,         d11                 \n\t"
      "vmlal.s16    q8,         d6,            d5[0]\n\t"
      "vmlal.s16    q9,         d7,            d5[0]\n\t"
      "vmlal.s16    q10,        d6,            d5[1]\n\t"
      "vmlal.s16    q11,        d7,            d5[1]\n\t"
      "vmlal.s16    q12,        d6,            d5[2]\n\t"
      "vmlal.s16    q13,        d7,            d5[2]\n\t"
      "vmlal.s16    q14,        d6,            d5[3]\n\t"
      "vmlal.s16    q15,        d7,            d5[3]\n\t"
      "2:                                          \n\t"  // last 2 rows
      "subs         %[kc5],     %[kc5],        #1  \n\t"
      "blt          3f                             \n\t"
      "vld1.s8      {d0},       [%[a_ptr]]!        \n\t"  // load A 2 cols
      "vld1.s8      {d8-d9},    [%[b_ptr]]!        \n\t"  // load B 2 rows
      "vmovl.s8     q2,         d0                 \n\t"
      "vmovl.s8     q3,         d8                 \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "vmovl.s8     q3,         d9                 \n\t"
      "vmlal.s16    q8,         d6,            d5[0]\n\t"
      "vmlal.s16    q9,         d7,            d5[0]\n\t"
      "vmlal.s16    q10,        d6,            d5[1]\n\t"
      "vmlal.s16    q11,        d7,            d5[1]\n\t"
      "vmlal.s16    q12,        d6,            d5[2]\n\t"
      "vmlal.s16    q13,        d7,            d5[2]\n\t"
      "vmlal.s16    q14,        d6,            d5[3]\n\t"
      "vmlal.s16    q15,        d7,            d5[3]\n\t"
      "3:                                          \n\t"  // last 1 row
      "subs         %[kc6],     %[kc6],        #1  \n\t"
      "blt          4f                             \n\t"
      "vld1.s8      {d0},       [%[a_ptr]]         \n\t"  // load A 1 col
      "vld1.s8      {d8},       [%[b_ptr]]        \n\t"   // load B 1 row
      "vmovl.s8     q2,         d0                 \n\t"
      "vmovl.s8     q3,         d8                 \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "4:                                          \n\t"
      "vst1.32      {q8, q9},   [%[c]],   %[step]  \n\t"
      "vst1.32      {q10, q11}, [%[c]],   %[step]  \n\t"
      "vst1.32      {q12, q13}, [%[c]],   %[step]  \n\t"
      "vst1.32      {q14, q15}, [%[c]]             \n\t"
      :
      : [a_ptr] "r"(a_ptr), [b_ptr] "r"(b_ptr), [c] "r"(c), [kc1] "r"(kc1),
        [kc3] "r"(kc3), [kc5] "r"(kc5), [kc6] "r"(kc6), [step] "r"(step)
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif  // __aarch64__
#endif  // __ARM_NEON
}

// The core idea of AddDot4x2 function is borrowed from the Google's gemmlowp
// open source library. The address of gemmlowp is
// https://github.com/google/gemmlowp.
void Gemm::AddDot4x2(int32_t k, const int8_t *a, const int8_t *b, int32_t *c,
                     int32_t ldc) {
#if __ARM_NEON
#if __aarch64__
// AddDot4x2 used only for aarch32
#else
#define PADDLE_LABEL_LOOP "1"
#define PADDLE_LABEL_AFTER_LOOP "2"
  asm volatile(
      "lsl %[ldc], %[ldc], #2 \n\t"  // sizeof(int32) == 4
      "vldr d0, [%[b], #0] \n\t"
      "vmov.s32 q8, #0 \n\t"
      "vldr d4, [%[a], #0] \n\t"
      "vmov.s32 q9, q8 \n\t"
      "vldr d2, [%[b], #16] \n\t"
      "vmov.s32 q10, q8 \n\t"
      "vldr d6, [%[a], #16] \n\t"
      "vmov.s32 q11, q8 \n\t"
      "vldr d1, [%[b], #8]\n\t"
      "vmov.s32 q12, q8 \n\t"
      "vldr d5, [%[a], #8]\n"
      "vmov.s32 q13, q8 \n\t"
      "vldr d3, [%[b], #24]\n\t"
      "vmov.s32 q14, q8 \n\t"
      "vldr d7, [%[a], #24]\n"
      "vmov.s32 q15, q8 \n\t"

      PADDLE_LABEL_LOOP
      ": \n\t"
      "vmull.s8    q4,  d0,  d4 \n\t"  // first half
      "add %[b], %[b], #32 \n\t"
      "vmull.s8    q5,  d2,  d4 \n\t"
      "vldr d4, [%[a], #32] \n\t"
      "vmull.s8    q6,  d0,  d6 \n\t"
      "vmull.s8    q7,  d2,  d6 \n\t"
      "vldr d6, [%[a], #48] \n\t"

      "vmlal.s8    q4,  d1,  d5 \n\t"  // second half
      "vmlal.s8    q5,  d3,  d5 \n\t"
      "vldr d5, [%[a], #40] \n\t"
      "vmlal.s8    q6,  d1,  d7 \n\t"
      "vmlal.s8    q7,  d3,  d7 \n\t"
      "vldr d7, [%[a], #56] \n\t"

      "vpadal.s16   q8,  q4 \n\t"  // pairwise-add
      "add %[a], %[a], #64 \n\t"
      "vpadal.s16   q9,  q5 \n\t"
      "subs %[k], %[k], #16 \n\t"
      "vpadal.s16   q10, q6 \n\t"
      "vpadal.s16   q11, q7 \n\t"

      "beq " PADDLE_LABEL_AFTER_LOOP
      "f \n\t"

      "vmull.s8    q4,  d0,  d4 \n\t"  // first half
      "vmull.s8    q5,  d2,  d4 \n\t"
      "vldr d4, [%[a], #0] \n\t"
      "vmull.s8    q6,  d0,  d6 \n\t"
      "vldr d0, [%[b], #0] \n\t"
      "vmull.s8    q7,  d2,  d6 \n\t"
      "vldr d2, [%[b], #16] \n\t"

      "vmlal.s8    q4,  d1,  d5 \n\t"  // second half
      "vldr d6, [%[a], #16] \n\t"
      "vmlal.s8    q5,  d3,  d5 \n\t"
      "vldr d5, [%[a], #8] \n\t"
      "vmlal.s8    q6,  d1,  d7 \n\t"
      "vldr d1, [%[b], #8] \n\t"
      "vmlal.s8    q7,  d3,  d7 \n\t"
      "vldr d3, [%[b], #24] \n\t"

      "vpadal.s16   q12, q4 \n\t"  // pairwise-add
      "vldr d7, [%[a], #24] \n\t"
      "vpadal.s16   q13, q5 \n\t"
      "vpadal.s16   q14, q6 \n\t"
      "vpadal.s16   q15, q7 \n\t"

      "b " PADDLE_LABEL_LOOP "b \n\t"

      PADDLE_LABEL_AFTER_LOOP
      ": \n\t"
      "vmull.s8    q4,  d0,  d4 \n\t"  // first half
      "vmull.s8    q5,  d2,  d4 \n\t"
      "vmull.s8    q6,  d0,  d6 \n\t"
      "vmull.s8    q7,  d2,  d6 \n\t"

      "vmlal.s8    q4,  d1,  d5 \n\t"  // second half
      "vmlal.s8    q5,  d3,  d5 \n\t"
      "vmlal.s8    q6,  d1,  d7 \n\t"
      "vmlal.s8    q7,  d3,  d7 \n\t"

      "vpadal.s16   q12, q4 \n\t"  // pairwise-add
      "vpadal.s16   q13, q5 \n\t"
      "vpadal.s16   q14, q6 \n\t"
      "vpadal.s16   q15, q7 \n\t"

      "vpadd.s32 d0, d16, d17 \n\t"  // reduce to int32
      "vpadd.s32 d1, d18, d19 \n\t"
      "vpadd.s32 d2, d20, d21 \n\t"
      "vpadd.s32 d3, d22, d23 \n\t"
      "vpadd.s32 d4, d24, d25 \n\t"
      "vpadd.s32 d5, d26, d27 \n\t"
      "vpadd.s32 d6, d28, d29 \n\t"
      "vpadd.s32 d7, d30, d31 \n\t"

      "vpadd.s32 d8, d0, d1 \n\t"  // reduce to int32 again
      "vpadd.s32 d9, d2, d3 \n\t"
      "vpadd.s32 d10, d4, d5 \n\t"
      "vpadd.s32 d11, d6, d7 \n\t"

      "vst1.32 {d8}, [%[c]], %[ldc] \n\t"
      "vst1.32 {d9}, [%[c]], %[ldc] \n\t"
      "vst1.32 {d10}, [%[c]], %[ldc] \n\t"
      "vst1.32 {d11}, [%[c]]  \n\t"

      : [k] "+r"(k), [a] "+r"(a), [b] "+r"(b), [c] "+r"(c)
      : [ldc] "r"(ldc)
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#undef PADDLE_LABEL_AFTER_LOOP
#undef PADDLE_LABEL_LOOP

#endif  // __aarch64__
#endif  // __ARM_NEON
}

void Gemm::AddDot4x4(int32_t k, const int8_t *a, const int8_t *b, int32_t *c,
                     int32_t ldc) {
#if __ARM_NEON
#if __aarch64__
#define PADDLE_LABEL_LOOP "1"
#define PADDLE_LABEL_AFTER_LOOP "2"
  asm volatile(
      // load data from matrix a and b，and set zero to result register
      "ld1 {v0.16b}, [%[b]], #16\n"
      "dup v16.4s, wzr\n"
      "ld1 {v4.16b}, [%[a]], #16\n"
      "dup v17.4s, wzr\n"
      "ld1 {v1.16b}, [%[b]], #16\n"
      "dup v18.4s, wzr\n"
      "ld1 {v5.16b}, [%[a]], #16\n"
      "dup v19.4s, wzr\n"
      "ld1 {v2.16b}, [%[b]], #16\n"
      "dup v20.4s, wzr\n"
      "ld1 {v3.16b}, [%[b]], #16\n"
      "dup v21.4s, wzr\n"
      "ld1 {v6.16b}, [%[a]], #16\n"
      "dup v22.4s, wzr\n"
      "ld1 {v7.16b}, [%[a]], #16\n"
      "dup v23.4s, wzr\n"
      "dup v24.4s, wzr\n"
      "dup v25.4s, wzr\n"
      "dup v26.4s, wzr\n"
      "dup v27.4s, wzr\n"
      "dup v28.4s, wzr\n"
      "dup v29.4s, wzr\n"
      "dup v30.4s, wzr\n"
      "dup v31.4s, wzr\n"

      // Multiply ldc by 4 == sizeof(int32)
      "lsl %[ldc], %[ldc], #2\n"

      // first half
      "smull    v8.8h,  v0.8b,  v4.8b\n"
      "smull    v9.8h,  v1.8b,  v4.8b\n"
      "smull    v10.8h,  v2.8b,  v4.8b\n"
      "smull    v11.8h,  v3.8b,  v4.8b\n"
      "smull    v12.8h,  v0.8b,  v5.8b\n"
      "smull    v13.8h,  v1.8b,  v5.8b\n"
      "smull    v14.8h,  v2.8b,  v5.8b\n"
      "smull    v15.8h,  v3.8b,  v5.8b\n"

      // Multiply-accumulate second-half
      "smlal2   v8.8h,  v0.16b,  v4.16b\n"
      "smlal2   v9.8h,  v1.16b,  v4.16b\n"
      "smlal2   v10.8h,  v2.16b,  v4.16b\n"
      "smlal2   v11.8h,  v3.16b,  v4.16b\n"
      "smlal2   v12.8h,  v0.16b,  v5.16b\n"
      "smlal2   v13.8h,  v1.16b,  v5.16b\n"
      "smlal2   v14.8h,  v2.16b,  v5.16b\n"
      "smlal2   v15.8h,  v3.16b,  v5.16b\n"

      "subs %[k], %[k], #16\n"

      // skip the loop
      "beq " PADDLE_LABEL_AFTER_LOOP "f\n"

      // loop
      PADDLE_LABEL_LOOP
      ":\n"

      // first half
      "sadalp  v16.4s, v8.8h\n"
      "ld1 {v4.16b}, [%[a]], #16\n"
      "smull    v8.8h,  v0.8b,  v6.8b\n"
      "sadalp  v17.4s, v9.8h\n"
      "ld1 {v5.16b}, [%[a]], #16\n"
      "smull    v9.8h,  v1.8b,  v6.8b\n"
      "sadalp  v18.4s, v10.8h\n"
      "smull    v10.8h,  v2.8b,  v6.8b\n"
      "sadalp  v19.4s, v11.8h\n"
      "smull    v11.8h,  v3.8b,  v6.8b\n"
      "sadalp  v20.4s, v12.8h\n"
      "smull    v12.8h,  v0.8b,  v7.8b\n"
      "sadalp  v21.4s, v13.8h\n"
      "smull    v13.8h,  v1.8b,  v7.8b\n"
      "sadalp  v22.4s, v14.8h\n"
      "smull    v14.8h,  v2.8b,  v7.8b\n"
      "sadalp  v23.4s, v15.8h\n"
      "smull    v15.8h,  v3.8b,  v7.8b\n"

      // Multiply-accumulate second-half
      "smlal2   v8.8h,  v0.16b,  v6.16b\n"
      "smlal2   v9.8h,  v1.16b,  v6.16b\n"
      "smlal2   v10.8h,  v2.16b,  v6.16b\n"
      "smlal2   v11.8h,  v3.16b,  v6.16b\n"

      "ld1 {v6.16b}, [%[a]], #16\n"

      "smlal2   v12.8h,  v0.16b,  v7.16b\n"
      "ld1 {v0.16b}, [%[b]], #16\n"
      "smlal2   v13.8h,  v1.16b,  v7.16b\n"
      "ld1 {v1.16b}, [%[b]], #16\n"
      "smlal2   v14.8h,  v2.16b,  v7.16b\n"
      "ld1 {v2.16b}, [%[b]], #16\n"
      "smlal2   v15.8h,  v3.16b,  v7.16b\n"
      "ld1 {v3.16b}, [%[b]], #16\n"

      // first half
      "sadalp  v24.4s, v8.8h\n"
      "smull    v8.8h,  v0.8b,  v4.8b\n"
      "sadalp  v25.4s, v9.8h\n"
      "ld1 {v7.16b}, [%[a]], #16\n"
      "smull    v9.8h,  v1.8b,  v4.8b\n"
      "sadalp  v26.4s, v10.8h\n"
      "smull    v10.8h,  v2.8b,  v4.8b\n"
      "sadalp  v27.4s, v11.8h\n"
      "smull    v11.8h,  v3.8b,  v4.8b\n"
      "sadalp  v28.4s, v12.8h\n"
      "smull    v12.8h,  v0.8b,  v5.8b\n"
      "sadalp  v29.4s, v13.8h\n"
      "smull    v13.8h,  v1.8b,  v5.8b\n"
      "sadalp  v30.4s, v14.8h\n"
      "smull    v14.8h,  v2.8b,  v5.8b\n"
      "sadalp  v31.4s, v15.8h\n"
      "smull    v15.8h,  v3.8b,  v5.8b\n"

      // Multiply-accumulate second-half
      "smlal2   v8.8h,  v0.16b,  v4.16b\n"
      "smlal2   v9.8h,  v1.16b,  v4.16b\n"
      "smlal2   v10.8h,  v2.16b,  v4.16b\n"
      "smlal2   v11.8h,  v3.16b,  v4.16b\n"

      // Loop
      "subs %[k], %[k], #16\n"

      "smlal2   v12.8h,  v0.16b,  v5.16b\n"
      "smlal2   v13.8h,  v1.16b,  v5.16b\n"
      "smlal2   v14.8h,  v2.16b,  v5.16b\n"
      "smlal2   v15.8h,  v3.16b,  v5.16b\n"

      "bne " PADDLE_LABEL_LOOP "b\n"

      // Final
      PADDLE_LABEL_AFTER_LOOP
      ":\n"

      // first half
      "sadalp  v16.4s, v8.8h\n"
      "smull    v8.8h,  v0.8b,  v6.8b\n"
      "sadalp  v17.4s, v9.8h\n"
      "smull    v9.8h,  v1.8b,  v6.8b\n"
      "sadalp  v18.4s, v10.8h\n"
      "smull    v10.8h,  v2.8b,  v6.8b\n"
      "sadalp  v19.4s, v11.8h\n"
      "smull    v11.8h,  v3.8b,  v6.8b\n"
      "sadalp  v20.4s, v12.8h\n"
      "smull    v12.8h,  v0.8b,  v7.8b\n"
      "sadalp  v21.4s, v13.8h\n"
      "smull    v13.8h,  v1.8b,  v7.8b\n"
      "sadalp  v22.4s, v14.8h\n"
      "smull    v14.8h,  v2.8b,  v7.8b\n"
      "sadalp  v23.4s, v15.8h\n"
      "smull    v15.8h,  v3.8b,  v7.8b\n"

      // Multiply-accumulate second-half
      "smlal2   v8.8h,  v0.16b,  v6.16b\n"
      "smlal2   v9.8h,  v1.16b,  v6.16b\n"
      "smlal2   v10.8h,  v2.16b,  v6.16b\n"
      "smlal2   v11.8h,  v3.16b,  v6.16b\n"
      "smlal2   v12.8h,  v0.16b,  v7.16b\n"
      "smlal2   v13.8h,  v1.16b,  v7.16b\n"
      "smlal2   v14.8h,  v2.16b,  v7.16b\n"
      "smlal2   v15.8h,  v3.16b,  v7.16b\n"

      "sadalp  v24.4s, v8.8h\n"
      "sadalp  v25.4s, v9.8h\n"
      "sadalp  v26.4s, v10.8h\n"
      "sadalp  v27.4s, v11.8h\n"
      "sadalp  v28.4s, v12.8h\n"
      "sadalp  v29.4s, v13.8h\n"
      "sadalp  v30.4s, v14.8h\n"
      "sadalp  v31.4s, v15.8h\n"

      // Reduce 32bit accumulators horizontally.
      "addp v0.4s, v16.4s, v17.4s\n"
      "addp v1.4s, v18.4s, v19.4s\n"
      "addp v2.4s, v20.4s, v21.4s\n"
      "addp v3.4s, v22.4s, v23.4s\n"
      "addp v4.4s, v24.4s, v25.4s\n"
      "addp v5.4s, v26.4s, v27.4s\n"
      "addp v6.4s, v28.4s, v29.4s\n"
      "addp v7.4s, v30.4s, v31.4s\n"

      // Reduce 32bit accumulators horizontally, second pass
      // (each pass adds pairwise. we need to add 4-wise).
      "addp v12.4s, v0.4s, v1.4s\n"
      "addp v13.4s, v2.4s, v3.4s\n"
      "addp v14.4s, v4.4s, v5.4s\n"
      "addp v15.4s, v6.4s, v7.4s\n"

      "st1 {v12.4s}, [%[c]], %[ldc] \n\t"
      "st1 {v13.4s}, [%[c]], %[ldc] \n\t"
      "st1 {v14.4s}, [%[c]], %[ldc] \n\t"
      "st1 {v15.4s}, [%[c]]  \n\t"

      : [k] "+r"(k), [a] "+r"(a), [b] "+r"(b), [c] "+r"(c)  // outputs
      : [ldc] "r"(ldc)                                      // inputs
      : "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31");  // clobbers
#undef PADDLE_LABEL_AFTER_LOOP
#undef PADDLE_LABEL_LOOP
#else
// AddDot4x2 used only for aarch64
#endif  // __aarch64__
#endif  // __ARM_NEON
}

// 8 bits int small block inner product
void Gemm::AddDot6x8(int32_t k, const int8_t *a, const int8_t *b, int32_t *c,
                     int32_t ldc) {
#if __ARM_NEON
#if __aarch64__
// AddDot6x8 used only for aarch32
#else
  const int8_t *a_ptr, *b_ptr;
  a_ptr = a;
  b_ptr = b;
  int32_t kc1 = k >> 3;
  int32_t kc2 = k & 7;
  int32_t kc3 = kc2 >> 2;
  int32_t kc4 = kc2 & 3;
  int32_t kc5 = kc4 >> 1;
  int32_t kc6 = kc4 & 1;
  int32_t step = sizeof(int32_t) * ldc;
  asm volatile(
      // q4-q15: save 48 results
      "pld          [%[a_ptr]]                     \n\t"
      "pld          [%[b_ptr]]                     \n\t"
      "pld          [%[b_ptr], #64]                \n\t"
      "vmov.s32     q4,         #0                 \n\t"
      "vmov.s32     q5,         q4                 \n\t"
      "vmov.s32     q6,         q4                 \n\t"
      "vmov.s32     q7,         q4                 \n\t"
      "vmov.s32     q8,         q4                 \n\t"
      "vmov.s32     q9,         q4                 \n\t"
      "vmov.s32     q10,        q4                 \n\t"
      "vmov.s32     q11,        q4                 \n\t"
      "vmov.s32     q12,        q4                 \n\t"
      "vmov.s32     q13,        q4                 \n\t"
      "vmov.s32     q14,        q4                 \n\t"
      "vmov.s32     q15,        q4                 \n\t"
      "mov r0,      #12                            \n\t"
      "subs         %[kc1],     %[kc1],       #1   \n\t"
      "blt          1f                             \n\t"
      "0:                                          \n\t"
      "pld          [%[a_ptr], #64]                \n\t"
      "pld          [%[b_ptr], #128]               \n\t"
      "vld1.s8      {d0-d2},    [%[a_ptr]]!        \n\t"  // A 4 cols
      "vld1.s8      {d3},       [%[b_ptr]]!        \n\t"  // B 1st row
      "vmovl.s8     q2,         d0                 \n\t"
      "vmovl.s8     q3,         d3                 \n\t"
      "vmlal.s16    q4,         d6,            d4[0]\n\t"
      "vmlal.s16    q5,         d7,            d4[0]\n\t"
      "vmlal.s16    q6,         d6,            d4[1]\n\t"
      "vmlal.s16    q7,         d7,            d4[1]\n\t"
      "vmlal.s16    q8,         d6,            d4[2]\n\t"
      "vmlal.s16    q9,         d7,            d4[2]\n\t"
      "vmlal.s16    q10,        d6,            d4[3]\n\t"
      "vmlal.s16    q11,        d7,            d4[3]\n\t"
      "vmlal.s16    q12,        d6,            d5[0]\n\t"
      "vmlal.s16    q13,        d7,            d5[0]\n\t"
      "vmlal.s16    q14,        d6,            d5[1]\n\t"
      "vmlal.s16    q15,        d7,            d5[1]\n\t"
      "vld1.s8      {d3},       [%[b_ptr]]!         \n\t"  // B 2nd row
      "vmovl.s8     q3,         d3                  \n\t"
      "vmlal.s16    q4,         d6,            d5[2]\n\t"
      "vmlal.s16    q5,         d7,            d5[2]\n\t"
      "vmlal.s16    q6,         d6,            d5[3]\n\t"
      "vmlal.s16    q7,         d7,            d5[3]\n\t"
      "vmovl.s8     q2,         d1                  \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "vld1.s8      {d3},       [%[b_ptr]]!         \n\t"  // B 3th row
      "vmovl.s8     q3,         d3                  \n\t"
      "vmlal.s16    q4,         d6,            d5[0]\n\t"
      "vmlal.s16    q5,         d7,            d5[0]\n\t"
      "vmlal.s16    q6,         d6,            d5[1]\n\t"
      "vmlal.s16    q7,         d7,            d5[1]\n\t"
      "vmlal.s16    q8,         d6,            d5[2]\n\t"
      "vmlal.s16    q9,         d7,            d5[2]\n\t"
      "vmlal.s16    q10,        d6,            d5[3]\n\t"
      "vmlal.s16    q11,        d7,            d5[3]\n\t"
      "vmovl.s8     q2,         d2                  \n\t"
      "vmlal.s16    q12,        d6,            d4[0]\n\t"
      "vmlal.s16    q13,        d7,            d4[0]\n\t"
      "vmlal.s16    q14,        d6,            d4[1]\n\t"
      "vmlal.s16    q15,        d7,            d4[1]\n\t"
      "vld1.s8      {d3},       [%[b_ptr]]!         \n\t"  // B 4th row
      "vmovl.s8     q3,         d3                  \n\t"
      "vmlal.s16    q4,         d6,            d4[2]\n\t"
      "vmlal.s16    q5,         d7,            d4[2]\n\t"
      "vmlal.s16    q6,         d6,            d4[3]\n\t"
      "vmlal.s16    q7,         d7,            d4[3]\n\t"
      "vmlal.s16    q8,         d6,            d5[0]\n\t"
      "vmlal.s16    q9,         d7,            d5[0]\n\t"
      "vmlal.s16    q10,        d6,            d5[1]\n\t"
      "vmlal.s16    q11,        d7,            d5[1]\n\t"
      "vmlal.s16    q12,        d6,            d5[2]\n\t"
      "vmlal.s16    q13,        d7,            d5[2]\n\t"
      "vmlal.s16    q14,        d6,            d5[3]\n\t"
      "vmlal.s16    q15,        d7,            d5[3]\n\t"

      "vld1.s8      {d0-d2},    [%[a_ptr]]!        \n\t"  // A 4 cols
      "vld1.s8      {d3},       [%[b_ptr]]!        \n\t"  // B 1st row
      "vmovl.s8     q2,         d0                 \n\t"
      "vmovl.s8     q3,         d3                 \n\t"
      "vmlal.s16    q4,         d6,            d4[0]\n\t"
      "vmlal.s16    q5,         d7,            d4[0]\n\t"
      "vmlal.s16    q6,         d6,            d4[1]\n\t"
      "vmlal.s16    q7,         d7,            d4[1]\n\t"
      "vmlal.s16    q8,         d6,            d4[2]\n\t"
      "vmlal.s16    q9,         d7,            d4[2]\n\t"
      "vmlal.s16    q10,        d6,            d4[3]\n\t"
      "vmlal.s16    q11,        d7,            d4[3]\n\t"
      "vmlal.s16    q12,        d6,            d5[0]\n\t"
      "vmlal.s16    q13,        d7,            d5[0]\n\t"
      "vmlal.s16    q14,        d6,            d5[1]\n\t"
      "vmlal.s16    q15,        d7,            d5[1]\n\t"
      "vld1.s8      {d3},       [%[b_ptr]]!         \n\t"  // B 2nd row
      "vmovl.s8     q3,         d3                  \n\t"
      "vmlal.s16    q4,         d6,            d5[2]\n\t"
      "vmlal.s16    q5,         d7,            d5[2]\n\t"
      "vmlal.s16    q6,         d6,            d5[3]\n\t"
      "vmlal.s16    q7,         d7,            d5[3]\n\t"
      "vmovl.s8     q2,         d1                  \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "vld1.s8      {d3},       [%[b_ptr]]!         \n\t"  // B 3th row
      "vmovl.s8     q3,         d3                  \n\t"
      "vmlal.s16    q4,         d6,            d5[0]\n\t"
      "vmlal.s16    q5,         d7,            d5[0]\n\t"
      "vmlal.s16    q6,         d6,            d5[1]\n\t"
      "vmlal.s16    q7,         d7,            d5[1]\n\t"
      "vmlal.s16    q8,         d6,            d5[2]\n\t"
      "vmlal.s16    q9,         d7,            d5[2]\n\t"
      "vmlal.s16    q10,        d6,            d5[3]\n\t"
      "vmlal.s16    q11,        d7,            d5[3]\n\t"
      "vmovl.s8     q2,         d2                  \n\t"
      "vmlal.s16    q12,        d6,            d4[0]\n\t"
      "vmlal.s16    q13,        d7,            d4[0]\n\t"
      "vmlal.s16    q14,        d6,            d4[1]\n\t"
      "vmlal.s16    q15,        d7,            d4[1]\n\t"
      "vld1.s8      {d3},       [%[b_ptr]]!         \n\t"  // B 4th row
      "vmovl.s8     q3,         d3                  \n\t"
      "vmlal.s16    q4,         d6,            d4[2]\n\t"
      "vmlal.s16    q5,         d7,            d4[2]\n\t"
      "vmlal.s16    q6,         d6,            d4[3]\n\t"
      "vmlal.s16    q7,         d7,            d4[3]\n\t"
      "vmlal.s16    q8,         d6,            d5[0]\n\t"
      "vmlal.s16    q9,         d7,            d5[0]\n\t"
      "vmlal.s16    q10,        d6,            d5[1]\n\t"
      "vmlal.s16    q11,        d7,            d5[1]\n\t"
      "vmlal.s16    q12,        d6,            d5[2]\n\t"
      "vmlal.s16    q13,        d7,            d5[2]\n\t"
      "vmlal.s16    q14,        d6,            d5[3]\n\t"
      "vmlal.s16    q15,        d7,            d5[3]\n\t"

      "subs         %[kc1],     %[kc1],        #1  \n\t"
      "bge          0b                             \n\t"
      "1:                                          \n\t"  // last <8 rows
      "subs         %[kc3],     %[kc3],        #1  \n\t"
      "blt          2f                             \n\t"
      "vld1.s8      {d0-d2},    [%[a_ptr]]!        \n\t"  // A 4 cols
      "vld1.s8      {d3},       [%[b_ptr]]!        \n\t"  // B 1st row
      "vmovl.s8     q2,         d0                 \n\t"
      "vmovl.s8     q3,         d3                 \n\t"
      "vmlal.s16    q4,         d6,            d4[0]\n\t"
      "vmlal.s16    q5,         d7,            d4[0]\n\t"
      "vmlal.s16    q6,         d6,            d4[1]\n\t"
      "vmlal.s16    q7,         d7,            d4[1]\n\t"
      "vmlal.s16    q8,         d6,            d4[2]\n\t"
      "vmlal.s16    q9,         d7,            d4[2]\n\t"
      "vmlal.s16    q10,        d6,            d4[3]\n\t"
      "vmlal.s16    q11,        d7,            d4[3]\n\t"
      "vmlal.s16    q12,        d6,            d5[0]\n\t"
      "vmlal.s16    q13,        d7,            d5[0]\n\t"
      "vmlal.s16    q14,        d6,            d5[1]\n\t"
      "vmlal.s16    q15,        d7,            d5[1]\n\t"
      "vld1.s8      {d3},       [%[b_ptr]]!         \n\t"  // B 2nd row
      "vmovl.s8     q3,         d3                  \n\t"
      "vmlal.s16    q4,         d6,            d5[2]\n\t"
      "vmlal.s16    q5,         d7,            d5[2]\n\t"
      "vmlal.s16    q6,         d6,            d5[3]\n\t"
      "vmlal.s16    q7,         d7,            d5[3]\n\t"
      "vmovl.s8     q2,         d1                  \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "vld1.s8      {d3},       [%[b_ptr]]!         \n\t"  // B 3th row
      "vmovl.s8     q3,         d3                  \n\t"
      "vmlal.s16    q4,         d6,            d5[0]\n\t"
      "vmlal.s16    q5,         d7,            d5[0]\n\t"
      "vmlal.s16    q6,         d6,            d5[1]\n\t"
      "vmlal.s16    q7,         d7,            d5[1]\n\t"
      "vmlal.s16    q8,         d6,            d5[2]\n\t"
      "vmlal.s16    q9,         d7,            d5[2]\n\t"
      "vmlal.s16    q10,        d6,            d5[3]\n\t"
      "vmlal.s16    q11,        d7,            d5[3]\n\t"
      "vmovl.s8     q2,         d2                  \n\t"
      "vmlal.s16    q12,        d6,            d4[0]\n\t"
      "vmlal.s16    q13,        d7,            d4[0]\n\t"
      "vmlal.s16    q14,        d6,            d4[1]\n\t"
      "vmlal.s16    q15,        d7,            d4[1]\n\t"
      "vld1.s8      {d3},       [%[b_ptr]]!         \n\t"  // B 4th row
      "vmovl.s8     q3,         d3                  \n\t"
      "vmlal.s16    q4,         d6,            d4[2]\n\t"
      "vmlal.s16    q5,         d7,            d4[2]\n\t"
      "vmlal.s16    q6,         d6,            d4[3]\n\t"
      "vmlal.s16    q7,         d7,            d4[3]\n\t"
      "vmlal.s16    q8,         d6,            d5[0]\n\t"
      "vmlal.s16    q9,         d7,            d5[0]\n\t"
      "vmlal.s16    q10,        d6,            d5[1]\n\t"
      "vmlal.s16    q11,        d7,            d5[1]\n\t"
      "vmlal.s16    q12,        d6,            d5[2]\n\t"
      "vmlal.s16    q13,        d7,            d5[2]\n\t"
      "vmlal.s16    q14,        d6,            d5[3]\n\t"
      "vmlal.s16    q15,        d7,            d5[3]\n\t"

      "2:                                          \n\t"  // last <4 rows
      "subs         %[kc5],     %[kc5],        #1  \n\t"
      "blt          3f                             \n\t"
      "vld1.s8      {d0, d1},   [%[a_ptr]],    r0  \n\t"
      "vld1.s8      {d3},       [%[b_ptr]]!        \n\t"  // B 1st row
      "vmovl.s8     q2,         d0                 \n\t"
      "vmovl.s8     q3,         d3                 \n\t"
      "vmlal.s16    q4,         d6,            d4[0]\n\t"
      "vmlal.s16    q5,         d7,            d4[0]\n\t"
      "vmlal.s16    q6,         d6,            d4[1]\n\t"
      "vmlal.s16    q7,         d7,            d4[1]\n\t"
      "vmlal.s16    q8,         d6,            d4[2]\n\t"
      "vmlal.s16    q9,         d7,            d4[2]\n\t"
      "vmlal.s16    q10,        d6,            d4[3]\n\t"
      "vmlal.s16    q11,        d7,            d4[3]\n\t"
      "vmlal.s16    q12,        d6,            d5[0]\n\t"
      "vmlal.s16    q13,        d7,            d5[0]\n\t"
      "vmlal.s16    q14,        d6,            d5[1]\n\t"
      "vmlal.s16    q15,        d7,            d5[1]\n\t"
      "vld1.s8      {d3},       [%[b_ptr]]!         \n\t"  // B 2nd row
      "vmovl.s8     q3,         d3                  \n\t"
      "vmlal.s16    q4,         d6,            d5[2]\n\t"
      "vmlal.s16    q5,         d7,            d5[2]\n\t"
      "vmlal.s16    q6,         d6,            d5[3]\n\t"
      "vmlal.s16    q7,         d7,            d5[3]\n\t"
      "vmovl.s8     q2,         d1                  \n\t"
      "vmlal.s16    q8,         d6,            d4[0]\n\t"
      "vmlal.s16    q9,         d7,            d4[0]\n\t"
      "vmlal.s16    q10,        d6,            d4[1]\n\t"
      "vmlal.s16    q11,        d7,            d4[1]\n\t"
      "vmlal.s16    q12,        d6,            d4[2]\n\t"
      "vmlal.s16    q13,        d7,            d4[2]\n\t"
      "vmlal.s16    q14,        d6,            d4[3]\n\t"
      "vmlal.s16    q15,        d7,            d4[3]\n\t"
      "3:                                          \n\t"  // last <2 rows
      "subs         %[kc6],     %[kc6],        #1  \n\t"
      "blt          4f                             \n\t"
      "vld1.s8      {d0},       [%[a_ptr]]         \n\t"
      "vld1.s8      {d3},       [%[b_ptr]]         \n\t"
      "vmovl.s8     q2,         d0                 \n\t"
      "vmovl.s8     q3,         d3                 \n\t"
      "vmlal.s16    q4,         d6,            d4[0]\n\t"
      "vmlal.s16    q5,         d7,            d4[0]\n\t"
      "vmlal.s16    q6,         d6,            d4[1]\n\t"
      "vmlal.s16    q7,         d7,            d4[1]\n\t"
      "vmlal.s16    q8,         d6,            d4[2]\n\t"
      "vmlal.s16    q9,         d7,            d4[2]\n\t"
      "vmlal.s16    q10,        d6,            d4[3]\n\t"
      "vmlal.s16    q11,        d7,            d4[3]\n\t"
      "vmlal.s16    q12,        d6,            d5[0]\n\t"
      "vmlal.s16    q13,        d7,            d5[0]\n\t"
      "vmlal.s16    q14,        d6,            d5[1]\n\t"
      "vmlal.s16    q15,        d7,            d5[1]\n\t"
      "4:                                          \n\t"
      "vst1.32      {q4, q5},   [%[c]],   %[step]  \n\t"
      "vst1.32      {q6, q7},   [%[c]],   %[step]  \n\t"
      "vst1.32      {q8, q9},   [%[c]],   %[step]  \n\t"
      "vst1.32      {q10, q11}, [%[c]],   %[step]  \n\t"
      "vst1.32      {q12, q13}, [%[c]],   %[step]  \n\t"
      "vst1.32      {q14, q15}, [%[c]]             \n\t"
      :
      : [a_ptr] "r"(a_ptr), [b_ptr] "r"(b_ptr), [c] "r"(c), [kc1] "r"(kc1),
        [kc3] "r"(kc3), [kc5] "r"(kc5), [kc6] "r"(kc6), [step] "r"(step)
      : "cc", "memory", "r0", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
        "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif  // __aarch64__
#endif  // __ARM_NEON
}

// 8 bits int inner product
template <>
void Gemm::InnerKernel(int32_t mc, int32_t nc, float alpha, const int8_t *a,
                       const int8_t *b, float beta, int32_t *c, int8_t *C,
                       int32_t ldc, bool relu) {}
template <>
void Gemm::InnerKernel(int32_t mc, int32_t nc, float alpha, const int8_t *a,
                       const int8_t *b, float beta, int32_t *c, int32_t *C,
                       int32_t ldc, bool relu) {
#pragma omp parallel for
  for (int32_t j = 0; j < nc; j += NR_INT8) {
    for (int32_t i = 0; i < mc; i += MR_INT8) {
#if __aarch64__
      AddDot4x4(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#else
      AddDot4x2(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#endif  // __aarch64__
    }
  }
  if (!relu) {
    WriteBasic(mc, nc, c, C, ldc);
    return;
  }
}

template <>
void Gemm::InnerKernelWithBias(int32_t mc, int32_t nc, float alpha,
                               const int8_t *a, const int8_t *b, float beta,
                               int32_t *c, int8_t *C, int32_t ldc, bool relu,
                               int32_t *bias, bool addOnRow) {
#pragma omp parallel for
  for (int32_t j = 0; j < nc; j += NR_INT8) {
    for (int32_t i = 0; i < mc; i += MR_INT8) {
#if __aarch64__
      AddDot4x4(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#else
      AddDot4x2(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
#endif  // __aarch64__
    }
  }
  if (relu) {
    WriteWithAddReluScale(mc, nc, c, C, ldc, bias, alpha);
    return;
  } else {
    if (addOnRow) {
      WriteWithAddScaleT(mc, nc, c, C, ldc, bias, alpha);
    } else {
      WriteWithAddScale(mc, nc, c, C, ldc, bias, alpha);
    }
  }
}

template <>
void Gemm::InnerKernelWithBias(int32_t mc, int32_t nc, float alpha,
                               const int8_t *a, const int8_t *b, float beta,
                               int32_t *c, int32_t *C, int32_t ldc, bool relu,
                               int32_t *bias, bool addOnRow) {}

// 8 bits int PackMatrixA_4r
void Gemm::PackMatrixA_4r(int32_t m, int32_t k, int32_t m_tail, const int8_t *A,
                          int32_t lda, int8_t *buffer) {
  const int8_t *a0, *a1, *a2, *a3;
  for (int32_t i = 0; i < m - m_tail; i += 4) {
    a0 = A + i * lda;
    a1 = A + (i + 1) * lda;
    a2 = A + (i + 2) * lda;
    a3 = A + (i + 3) * lda;
    for (int32_t j = 0; j < k; ++j) {
      *buffer++ = *a0++;
      *buffer++ = *a1++;
      *buffer++ = *a2++;
      *buffer++ = *a3++;
    }
  }

  if (m_tail != 0) {
    a0 = &A(m - m_tail, 0);
    a1 = a0 + lda;
    a2 = a0 + 2 * lda;
    a3 = a0 + 3 * lda;
    switch (m_tail) {
      case 1:
        a1 = zero_int8;
      case 2:
        a2 = zero_int8;
      case 3:
        a3 = zero_int8;
        break;
      default:
        break;
    }
    for (int j = 0; j < k; ++j) {
      *buffer++ = *a0++;
      *buffer++ = *a1++;
      *buffer++ = *a2++;
      *buffer++ = *a3++;
    }
  }
}

// 8 bits int PackMatrixA_6r
void Gemm::PackMatrixA_6r(int32_t m, int32_t k, int32_t m_tail, const int8_t *A,
                          int32_t lda, int8_t *buffer) {
  const int32_t i_length = m - m_tail;
  for (int32_t i = 0; i < i_length; i += 6) {
    const int8_t *a0 = A + i * lda;
    const int8_t *a1 = A + (i + 1) * lda;
    const int8_t *a2 = A + (i + 2) * lda;
    const int8_t *a3 = A + (i + 3) * lda;
    const int8_t *a4 = A + (i + 4) * lda;
    const int8_t *a5 = A + (i + 5) * lda;
    int8_t *local_buffer = buffer + i * k;
    for (int32_t j = 0; j < k; ++j) {
      *local_buffer++ = *a0++;
      *local_buffer++ = *a1++;
      *local_buffer++ = *a2++;
      *local_buffer++ = *a3++;
      *local_buffer++ = *a4++;
      *local_buffer++ = *a5++;
    }
  }
  if (m_tail != 0) {
    const int8_t *a0 = &A(i_length, 0);
    const int8_t *a1 = a0 + lda;
    const int8_t *a2 = a0 + 2 * lda;
    const int8_t *a3 = a0 + 3 * lda;
    const int8_t *a4 = a0 + 4 * lda;
    const int8_t *a5 = a0 + 5 * lda;
    int8_t *local_buffer = buffer + i_length * k;
    switch (m_tail) {
      case 1:
        a1 = zero_int8;
      case 2:
        a2 = zero_int8;
      case 3:
        a3 = zero_int8;
      case 4:
        a4 = zero_int8;
      case 5:
        a5 = zero_int8;
        break;
      default:
        break;
    }
    for (int32_t j = 0; j < k; ++j) {
      *local_buffer++ = *a0++;
      *local_buffer++ = *a1++;
      *local_buffer++ = *a2++;
      *local_buffer++ = *a3++;
      *local_buffer++ = *a4++;
      *local_buffer++ = *a5++;
    }
  }
}

// 8 bits int PackMatrixB
void Gemm::PackMatrixB_8c(int32_t k, int32_t n, int32_t n_tail, const int8_t *B,
                          int32_t ldb, int8_t *buffer) {
  const int32_t j_length = n - n_tail;
  for (int32_t j = 0; j < j_length; j += 8) {
    int8_t *local_buffer = buffer + j * k;
    for (int32_t i = 0; i < k; ++i) {
      const int8_t *b0 = &B(i, j);
#if __ARM_NEON
#if __aarch64__
// PackMatrixB_8c used only for aarch32
#else
      asm volatile(
          //          "pld        [%[b0]]                     \n\t"
          "vld1.s8    {d0},   [%[b0]]         \n\t"
          "vst1.s8    {d0},   [%[local_buffer]]!    \n\t"
          : [local_buffer] "+r"(local_buffer)
          : [b0] "r"(b0)
          : "memory", "q0");
#endif  // __aarch64__
#else
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
      *local_buffer++ = *b0++;
#endif  // __ARM_NEON
    }
  }
  if (n_tail != 0) {
    int8_t *local_buffer = buffer + j_length * k;
    for (int32_t i = 0; i < k; ++i) {
      const int8_t *b0 = &B(i, j_length);
      for (int32_t j = j_length; j < n; ++j) {
        *local_buffer++ = *b0++;
      }
      for (int32_t j = n; j < j_length + 8; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
}

// 8 bits int PackMatrixA_4r
void Gemm::PackMatrixA_4r_16(int32_t m, int32_t k, int32_t m_tail,
                             const int8_t *A, int32_t lda, int8_t *buffer) {
  const int32_t i_length = m - m_tail;
  const int32_t k_count = k >> 4;
  const int32_t k_tail = k & 15;

  for (int32_t i = 0; i < i_length; i += 4) {
    const int8_t *a0 = A + i * lda;
    const int8_t *a1 = A + (i + 1) * lda;
    const int8_t *a2 = A + (i + 2) * lda;
    const int8_t *a3 = A + (i + 3) * lda;
    int8_t *local_buffer = buffer + i * KC;
    for (int32_t j = 0; j < k_count; ++j) {
#if __ARM_NEON
#if __aarch64__
      asm volatile(
          "ld1        {v0.16b},   [%[a0]],  #16    \n\t"
          "ld1        {v1.16b},   [%[a1]],  #16    \n\t"
          "ld1        {v2.16b},   [%[a2]],  #16    \n\t"
          "ld1        {v3.16b},   [%[a3]],  #16    \n\t"
          "st1        {v0.16b},   [%[local_buffer]],  #16   \n\t"
          "st1        {v1.16b},   [%[local_buffer]],  #16   \n\t"
          "st1        {v2.16b},   [%[local_buffer]],  #16   \n\t"
          "st1        {v3.16b},   [%[local_buffer]],  #16   \n\t"
          : [local_buffer] "+r"(local_buffer), [a0] "+r"(a0), [a1] "+r"(a1),
            [a2] "+r"(a2), [a3] "+r"(a3)
          :
          : "memory", "v0", "v1", "v2", "v3");
#else
      asm volatile(
          "vld1.s8    {d0, d1},   [%[a0]]!         \n\t"
          "vld1.s8    {d2, d3},   [%[a1]]!         \n\t"
          "vld1.s8    {d4, d5},   [%[a2]]!         \n\t"
          "vld1.s8    {d6, d7},   [%[a3]]!         \n\t"
          "vst1.s8    {d0, d1},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d2, d3},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d4, d5},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d6, d7},   [%[local_buffer]]!    \n\t"
          : [local_buffer] "+r"(local_buffer), [a0] "+r"(a0), [a1] "+r"(a1),
            [a2] "+r"(a2), [a3] "+r"(a3)
          :
          : "memory", "q0", "q1", "q2", "q3");
#endif  // __aarch64__
#else
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a0++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a1++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a2++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a3++;
      }
#endif  // __ARM_NEON
    }
    if (k_tail != 0) {
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a0++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }

      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a1++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a2++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a3++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
    }
  }

  if (m_tail != 0) {
    const int8_t *a0 = &A(i_length, 0);
    const int8_t *a1 = a0 + lda;
    const int8_t *a2 = a0 + 2 * lda;
    const int8_t *a3 = a0 + 3 * lda;
    int8_t *local_buffer = buffer + i_length * KC;
    switch (m_tail) {
      case 1:
        a1 = zero_int8;
      case 2:
        a2 = zero_int8;
      case 3:
        a3 = zero_int8;
        break;
      default:
        break;
    }
    for (int32_t j = 0; j < k_count; ++j) {
#if __ARM_NEON
#if __aarch64__
      asm volatile(
          "ld1        {v0.16b},   [%[a0]],  #16    \n\t"
          "ld1        {v1.16b},   [%[a1]],  #16    \n\t"
          "ld1        {v2.16b},   [%[a2]],  #16    \n\t"
          "ld1        {v3.16b},   [%[a3]],  #16    \n\t"
          "st1        {v0.16b},   [%[local_buffer]],  #16   \n\t"
          "st1        {v1.16b},   [%[local_buffer]],  #16   \n\t"
          "st1        {v2.16b},   [%[local_buffer]],  #16   \n\t"
          "st1        {v3.16b},   [%[local_buffer]],  #16   \n\t"
          : [local_buffer] "+r"(local_buffer), [a0] "+r"(a0), [a1] "+r"(a1),
            [a2] "+r"(a2), [a3] "+r"(a3)
          :
          : "memory", "v0", "v1", "v2", "v3");
#else
      asm volatile(
          "vld1.s8    {d0, d1},   [%[a0]]!         \n\t"
          "vld1.s8    {d2, d3},   [%[a1]]!         \n\t"
          "vld1.s8    {d4, d5},   [%[a2]]!         \n\t"
          "vld1.s8    {d6, d7},   [%[a3]]!         \n\t"
          "vst1.s8    {d0, d1},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d2, d3},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d4, d5},   [%[local_buffer]]!    \n\t"
          "vst1.s8    {d6, d7},   [%[local_buffer]]!    \n\t"
          : [local_buffer] "+r"(local_buffer), [a0] "+r"(a0), [a1] "+r"(a1),
            [a2] "+r"(a2), [a3] "+r"(a3)
          :
          : "memory", "q0", "q1", "q2", "q3");
#endif  // __aarch64__
#else
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a0++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a1++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a2++;
      }
      for (int32_t l = 0; l < 16; ++l) {
        *local_buffer++ = *a3++;
      }
#endif  // __ARM_NEON
    }
    if (k_tail != 0) {
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a0++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }

      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a1++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a2++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *a3++;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
}

// 8 bits int PackMatrixB
void Gemm::PackMatrixB_2c_16(int32_t k, int32_t n, int32_t n_tail,
                             const int8_t *B, int32_t ldb, int8_t *buffer) {
  const int32_t j_length = n - n_tail;
  const int32_t k_count = k >> 4;
  const int32_t k_tail = k & 15;
  for (int32_t j = 0; j < j_length; j += 2) {
    int8_t *local_buffer = buffer + j * KC;
    for (int32_t i = 0; i < k_count; ++i) {
      const int8_t *b0 = &B((i << 4), j);
      const int8_t *b1 = &B((i << 4), j + 1);
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = *b0;
        b0 += ldb;
      }
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = *b1;
        b1 += ldb;
      }
    }
    if (k_tail != 0) {
      const int8_t *b0 = &B((k_count << 4), j);
      const int8_t *b1 = &B((k_count << 4), j + 1);
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *b0;
        b0 += ldb;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }

      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *b1;
        b1 += ldb;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
  if (n_tail != 0) {
    int8_t *local_buffer = buffer + j_length * KC;
    for (int32_t i = 0; i < k_count; ++i) {
      const int8_t *b0 = &B((i << 4), j_length);
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = *b0;
        b0 += ldb;
      }
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = 0;
      }
    }
    if (k_tail != 0) {
      const int8_t *b0 = &B((k_count << 4), j_length);
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *b0;
        b0 += ldb;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
      for (int32_t j = k_count << 4; j < KC; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
}

void Gemm::PackMatrixB_4c_16(int32_t k, int32_t n, int32_t n_tail,
                             const int8_t *B, int32_t ldb, int8_t *buffer) {
  const int32_t j_length = n - n_tail;
  const int32_t k_count = k >> 4;
  const int32_t k_tail = k & 15;
  for (int32_t j = 0; j < n; j += 4) {
    int8_t *local_buffer = buffer + j * KC;
    const int8_t *b0 = &B(0, j);
    const int8_t *b1 = b0 + 1;
    const int8_t *b2 = b0 + 2;
    const int8_t *b3 = b0 + 3;
    if (j > j_length) {
      switch (n_tail) {
        case 1:
          b1 = zero_int8;
        case 2:
          b2 = zero_int8;
        case 3:
          b3 = zero_int8;
          break;
        default:
          break;
      }
    }

    for (int32_t i = 0; i < k_count; ++i) {
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = *b0;
        b0 += ldb;
      }
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = *b1;
        b1 += ldb;
      }
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = *b2;
        b2 += ldb;
      }
      for (int m = 0; m < 16; ++m) {
        *local_buffer++ = *b3;
        b3 += ldb;
      }
    }
    if (k_tail != 0) {
      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *b0;
        b0 += ldb;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }

      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *b1;
        b1 += ldb;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }

      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *b2;
        b2 += ldb;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }

      for (int32_t j = k_count << 4; j < k; ++j) {
        *local_buffer++ = *b3;
        b3 += ldb;
      }
      for (int32_t j = k; j < KC; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
}

//  8 bits int write back
// C = A * B
void Gemm::WriteBasic(int32_t mc, int32_t nc, int32_t *c, int32_t *C,
                      int32_t ldc) {
#if __ARM_NEON
#if __aarch64__
  int32_t nc1 = nc / 4;
  int32_t _nc1 = nc % 4;

  int32_t *c_ptr, *C_ptr;
  int32x4_t cv;
  for (int32_t i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    for (int32_t j = 0; j < nc1; ++j) {
      cv = vld1q_s32(c_ptr);
      vst1q_s32(C_ptr, cv);
      c_ptr += 4;
      C_ptr += 4;
    }
    if (_nc1 != 0) {
      cv = vld1q_s32(c_ptr);
      if (_nc1 >= 1) {
        vst1q_lane_s32(C_ptr, cv, 0);
        C_ptr++;
      }
      if (_nc1 >= 2) {
        vst1q_lane_s32(C_ptr, cv, 1);
        C_ptr++;
      }
      if (_nc1 >= 3) {
        vst1q_lane_s32(C_ptr, cv, 2);
      }
    }
  }
#else
  int32_t nc1 = nc >> 4;
  int32_t _nc1 = nc & 15;
  int32_t step = sizeof(int32_t) * ldc;
  int32_t step1 = sizeof(int32_t) * (NC - (nc1 << 4));
  int32_t volatile m = mc;
  int32_t volatile n = nc1;
  int32_t *volatile c_ptr, *volatile C_ptr;
  int32_t *C0, *c0;
  c_ptr = c;
  C_ptr = C;
  if (nc1 > 0) {
    asm volatile(
        "subs       %[mc], %[mc], #1        \n\t"
        "blt        end_mc_%=               \n\t"
        "loop_mc_%=:                        \n\t"

        "mov        r6,   %[C_ptr]          \n\t"
        "mov        r5,   %[nc1]            \n\t"
        "subs       r5,   r5,   #1          \n\t"
        "blt        end_nc1_%=              \n\t"
        "loop_nc1_%=:                       \n\t"

        "vld1.32    {q0, q1}, [%[c_ptr]]!   \n\t"
        "vst1.32    {q0, q1}, [r6]!         \n\t"

        "vld1.32    {q2, q3}, [%[c_ptr]]!   \n\t"
        "vst1.32    {q2, q3}, [r6]!         \n\t"

        "subs       r5,   r5,   #1          \n\t"
        "bge        loop_nc1_%=             \n\t"
        "end_nc1_%=:                        \n\t"

        "add        %[C_ptr], %[C_ptr], %[step]   \n\t"
        "add        %[c_ptr], %[c_ptr], %[step1]  \n\t"
        "subs       %[mc], %[mc], #1        \n\t"
        "bge        loop_mc_%=              \n\t"
        "end_mc_%=:                         \n\t"

        :
        : [C_ptr] "r"(C_ptr), [c_ptr] "r"(c_ptr), [mc] "r"(m), [nc1] "r"(n),
          [step] "r"(step), [step1] "r"(step1)
        : "memory", "r5", "r6", "q0", "q1", "q2", "q3");
  }

  if (_nc1 != 0) {
    for (int32_t i = 0; i < mc; i++) {
      C0 = C_ptr + nc1 * 16 + i * ldc;
      c0 = c_ptr + nc1 * 16 + i * NC;
      for (int32_t j = 0; j < _nc1; j++) {
        *C0++ = *c0++;
      }
    }
  }
#endif  // __aarch64__
#endif  // __ARM_NEON
}

// C = A * B + bias, scale * C, bias is added on column
void Gemm::WriteWithAddScale(int32_t mc, int32_t nc, int32_t *c, int8_t *C,
                             int32_t ldc, int32_t *bias, float scale) {
#if __ARM_NEON
#if __aarch64__
  int32_t nc1 = nc >> 3;
  int32_t _nc1 = nc & 7;

  int32_t *c_ptr;
  int8_t *C_ptr;
  int32x4_t cv0;
  int32x4_t cv1;
  int16x8_t cv_h;
  int8x8_t cv_b;
  int32x4_t biasv;
  int8_t min = -127;
  int8x8_t minv = vdup_n_s8(min);
  for (int32_t i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    biasv = vld1q_dup_s32(bias + i);
    for (int32_t j = 0; j < nc1; ++j) {
      cv0 = vld1q_s32(c_ptr);
      cv1 = vld1q_s32(c_ptr + 4);
      cv0 = vqaddq_s32(cv0, biasv);
      cv1 = vqaddq_s32(cv1, biasv);

      cv_h = vcombine_s16(vqmovn_s32(cv0), vqmovn_s32(cv1));
      cv_b = vqmovn_s16(cv_h);

      cv_b = vmax_s8(cv_b, minv);
      vst1_s8(C_ptr, cv_b);
      c_ptr += 8;
      C_ptr += 8;
    }
    if (_nc1 != 0) {
      cv0 = vld1q_s32(c_ptr);
      cv1 = vld1q_s32(c_ptr + 4);
      cv0 = vqaddq_s32(cv0, biasv);
      cv1 = vqaddq_s32(cv1, biasv);

      cv_h = vcombine_s16(vqmovn_s32(cv0), vqmovn_s32(cv1));
      cv_b = vqmovn_s16(cv_h);

      cv_b = vmax_s8(cv_b, minv);

      switch (_nc1) {
        case 7:
          vst1_lane_s8(C_ptr + 6, cv_b, 6);
        case 6:
          vst1_lane_s8(C_ptr + 5, cv_b, 5);
        case 5:
          vst1_lane_s8(C_ptr + 4, cv_b, 4);
        case 4:
          vst1_lane_s8(C_ptr + 3, cv_b, 3);
        case 3:
          vst1_lane_s8(C_ptr + 2, cv_b, 2);
        case 2:
          vst1_lane_s8(C_ptr + 1, cv_b, 1);
        case 1:
          vst1_lane_s8(C_ptr, cv_b, 0);
        default:
          break;
      }
    }
  }
#else
  int8_t narrow = -128;
  int32_t nc1 = nc >> 3;
  int32_t _nc1 = nc & 7;
  int32_t step = sizeof(int8_t) * ldc;
  int32_t step1 = sizeof(int32_t) * (NC - (nc1 << 3));
  int32_t volatile m = mc;
  int32_t volatile n = nc1;
  int32_t *volatile c_ptr, *volatile bias_ptr;
  int8_t *volatile C_ptr;
  c_ptr = c;
  C_ptr = C;
  bias_ptr = bias;
  if (nc1 > 0) {
    asm volatile(
        "subs       %[mc], %[mc], #1        \n\t"
        "blt        end_mc_%=               \n\t"
        "vdup.32    q15,  %[scale]          \n\t"
        "vdup.8     d24,  %[narrow]         \n\t"
        "loop_mc_%=:                        \n\t"
        "vld1.32    {d26[0]}, [%[bias_ptr]]!\n\t"
        "vdup.32    q13,  d26[0]            \n\t"
        "mov        r6,   %[C_ptr]          \n\t"
        "mov        r5,   %[nc1]            \n\t"
        "subs       r5,   r5,   #1          \n\t"
        "blt        end_nc1_%=              \n\t"
        "loop_nc1_%=:                       \n\t"
        "vld1.32    {q0, q1}, [%[c_ptr]]!   \n\t"
        "vqadd.s32  q0, q0, q13             \n\t"
        "vqadd.s32  q1, q1, q13             \n\t"
        "vcvt.f32.s32 q2, q0                \n\t"
        "vcvt.f32.s32 q3, q1                \n\t"
        "vmul.f32   q2, q2, q15             \n\t"
        "vmul.f32   q3, q3, q15             \n\t"
        "vcvt.s32.f32 q4, q2                \n\t"
        "vcvt.s32.f32 q5, q3                \n\t"
        "vqmovn.s32 d12, q4                 \n\t"
        "vqmovn.s32 d13, q5                 \n\t"
        "vqmovn.s16 d14, q6                 \n\t"
        "vceq.s8    d15, d14, d24           \n\t"
        "vsub.s8    d14, d14, d15           \n\t"
        "vst1.8     {d14}, [r6]!            \n\t"
        "subs       r5,   r5,   #1          \n\t"
        "bge        loop_nc1_%=             \n\t"
        "end_nc1_%=:                        \n\t"

        "add        %[C_ptr], %[C_ptr], %[step]  \n\t"
        "add        %[c_ptr], %[c_ptr], %[step1] \n\t"
        "subs       %[mc], %[mc], #1        \n\t"
        "bge        loop_mc_%=              \n\t"
        "end_mc_%=:                         \n\t"

        :
        : [C_ptr] "r"(C_ptr), [c_ptr] "r"(c_ptr), [mc] "r"(m), [nc1] "r"(n),
          [step] "r"(step), [step1] "r"(step1), [bias_ptr] "r"(bias_ptr),
          [scale] "r"(scale), [narrow] "r"(narrow)
        : "cc", "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
          "q7", "q12", "q13", "q15");
  }

  int32_t nc_left;
  int32_t *c0;
  int8_t *C0;
  int32_t bias_v;
  if (_nc1 != 0) {
    for (int32_t i = 0; i < mc; i++) {
      C0 = C_ptr + nc1 * 8 + i * ldc;
      c0 = c_ptr + nc1 * 8 + i * NC;
      bias_v = *(bias_ptr + i);
      nc_left = _nc1;
      asm volatile(
          "vdup.32    q15,  %[scale]          \n\t"
          "vdup.8     d24,  %[narrow]         \n\t"
          "vdup.32    q13,  %[bias_v]         \n\t"
          "cmp        %[_nc1], #4             \n\t"
          "blt        less_four_%=            \n\t"
          "vld1.32    {q0}, [%[c0]]!          \n\t"
          "vqadd.s32  q0, q0, q13             \n\t"
          "vcvt.f32.s32 q1, q0                \n\t"
          "vmul.f32   q1, q1, q15             \n\t"
          "vcvt.s32.f32 q2, q1                \n\t"
          "vqmovn.s32 d6, q2                  \n\t"
          "vqmovn.s16 d8, q3                  \n\t"
          "vceq.s8    d9, d8, d24             \n\t"
          "vsub.s8    d8, d8, d9              \n\t"
          "vst1.8     {d8[0]}, [%[C0]]!       \n\t"
          "vst1.8     {d8[1]}, [%[C0]]!       \n\t"
          "vst1.8     {d8[2]}, [%[C0]]!       \n\t"
          "vst1.8     {d8[3]}, [%[C0]]!       \n\t"
          "subs       %[_nc1], %[_nc1], #4    \n\t"
          "beq        process_over_%=         \n\t"
          "less_four_%=:                      \n\t"
          "vld1.32    {q0}, [%[c0]]          \n\t"
          "vqadd.s32  q0, q0, q13             \n\t"
          "vcvt.f32.s32 q1, q0                \n\t"
          "vmul.f32   q1, q1, q15             \n\t"
          "vcvt.s32.f32 q2, q1                \n\t"
          "vqmovn.s32 d6, q2                  \n\t"
          "vqmovn.s16 d8, q3                  \n\t"
          "vceq.s8    d9, d8, d24             \n\t"
          "vsub.s8    d8, d8, d9              \n\t"
          "loop_save_%=:                      \n\t"
          "vst1.8     {d8[0]}, [%[C0]]!       \n\t"
          "vext.8 d8, d8, d8, #1              \n\t"
          "subs       %[_nc1], %[_nc1], #1    \n\t"
          "bgt        loop_save_%=            \n\t"
          "process_over_%=:                   \n\t"
          :
          : [_nc1] "r"(nc_left), [C0] "r"(C0), [c0] "r"(c0),
            [bias_v] "r"(bias_v), [scale] "r"(scale), [narrow] "r"(narrow)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q12", "q13", "q15");
    }
  }
#endif  // __aarch64__
#endif  // __ARM_NEON
}

// C = A * B + bias, scale * C, bias is added on row
void Gemm::WriteWithAddScaleT(int32_t mc, int32_t nc, int32_t *c, int8_t *C,
                              int32_t ldc, int32_t *bias, float scale) {
#if __ARM_NEON
#if __aarch64__
  int32_t nc1 = nc >> 3;
  int32_t _nc1 = nc & 7;

  int32_t *c_ptr;
  int8_t *C_ptr;
  int32x4_t cv0;
  int32x4_t cv1;
  int16x8_t cv_h;
  int8x8_t cv_b;
  int32_t *bias_ptr;
  int32x4_t biasv0;
  int32x4_t biasv1;
  int8_t min = -127;
  int8x8_t minv = vdup_n_s8(min);
  for (int32_t i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    bias_ptr = bias;
    for (int32_t j = 0; j < nc1; ++j) {
      cv0 = vld1q_s32(c_ptr);
      cv1 = vld1q_s32(c_ptr + 4);
      biasv0 = vld1q_s32(bias_ptr);
      biasv1 = vld1q_s32(bias_ptr + 4);
      cv0 = vqaddq_s32(cv0, biasv0);
      cv1 = vqaddq_s32(cv1, biasv1);

      cv_h = vcombine_s16(vqmovn_s32(cv0), vqmovn_s32(cv1));
      cv_b = vqmovn_s16(cv_h);

      cv_b = vmax_s8(cv_b, minv);
      vst1_s8(C_ptr, cv_b);
      c_ptr += 8;
      C_ptr += 8;
      bias_ptr += 8;
    }
    if (_nc1 != 0) {
      cv0 = vld1q_s32(c_ptr);
      cv1 = vld1q_s32(c_ptr + 4);
      biasv0 = vld1q_s32(bias_ptr);
      biasv1 = vld1q_s32(bias_ptr + 4);
      cv0 = vqaddq_s32(cv0, biasv0);
      cv1 = vqaddq_s32(cv1, biasv1);

      cv_h = vcombine_s16(vqmovn_s32(cv0), vqmovn_s32(cv1));
      cv_b = vqmovn_s16(cv_h);

      cv_b = vmax_s8(cv_b, minv);

      switch (_nc1) {
        case 7:
          vst1_lane_s8(C_ptr + 6, cv_b, 6);
        case 6:
          vst1_lane_s8(C_ptr + 5, cv_b, 5);
        case 5:
          vst1_lane_s8(C_ptr + 4, cv_b, 4);
        case 4:
          vst1_lane_s8(C_ptr + 3, cv_b, 3);
        case 3:
          vst1_lane_s8(C_ptr + 2, cv_b, 2);
        case 2:
          vst1_lane_s8(C_ptr + 1, cv_b, 1);
        case 1:
          vst1_lane_s8(C_ptr, cv_b, 0);
        default:
          break;
      }
    }
  }
#else
  int8_t narrow = -128;
  int32_t nc1 = nc >> 3;
  int32_t _nc1 = nc & 7;
  int32_t step = sizeof(int8_t) * ldc;
  int32_t step1 = sizeof(int32_t) * (NC - (nc1 << 3));
  int32_t volatile m = mc;
  int32_t volatile n = nc1;
  int32_t *volatile c_ptr, *volatile bias_ptr;
  int8_t *volatile C_ptr;
  c_ptr = c;
  C_ptr = C;
  bias_ptr = bias;
  if (nc1 > 0) {
    asm volatile(
        "subs       %[mc], %[mc], #1        \n\t"
        "blt        end_mc_%=               \n\t"
        "vdup.32    q15,  %[scale]          \n\t"
        "vdup.8     d24,  %[narrow]         \n\t"
        "loop_mc_%=:                        \n\t"
        "mov        r4,   %[bias_ptr]       \n\t"
        "mov        r6,   %[C_ptr]          \n\t"
        "mov        r5,   %[nc1]            \n\t"
        "subs       r5,   r5,   #1          \n\t"
        "blt        end_nc1_%=              \n\t"
        "loop_nc1_%=:                       \n\t"
        "vld1.32    {q13, q14}, [r4]!        \n\t"
        "vld1.32    {q0, q1}, [%[c_ptr]]!   \n\t"
        "vqadd.s32  q0, q0, q13             \n\t"
        "vqadd.s32  q1, q1, q14             \n\t"
        "vcvt.f32.s32 q2, q0                \n\t"
        "vcvt.f32.s32 q3, q1                \n\t"
        "vmul.f32   q2, q2, q15             \n\t"
        "vmul.f32   q3, q3, q15             \n\t"
        "vcvt.s32.f32 q4, q2                \n\t"
        "vcvt.s32.f32 q5, q3                \n\t"
        "vqmovn.s32 d12, q4                 \n\t"
        "vqmovn.s32 d13, q5                 \n\t"
        "vqmovn.s16 d14, q6                 \n\t"
        "vceq.s8    d15, d14, d24           \n\t"
        "vsub.s8    d14, d14, d15           \n\t"
        "vst1.8     {d14}, [r6]!            \n\t"
        "subs       r5,   r5,   #1          \n\t"
        "bge        loop_nc1_%=             \n\t"
        "end_nc1_%=:                        \n\t"

        "add        %[C_ptr], %[C_ptr], %[step]  \n\t"
        "add        %[c_ptr], %[c_ptr], %[step1] \n\t"
        "subs       %[mc], %[mc], #1        \n\t"
        "bge        loop_mc_%=              \n\t"
        "end_mc_%=:                         \n\t"

        :
        : [C_ptr] "r"(C_ptr), [c_ptr] "r"(c_ptr), [mc] "r"(m), [nc1] "r"(n),
          [step] "r"(step), [step1] "r"(step1), [bias_ptr] "r"(bias_ptr),
          [scale] "r"(scale), [narrow] "r"(narrow)
        : "cc", "memory", "r4", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5",
          "q6", "q7", "q12", "q13", "q15");
  }

  int32_t nc_left;
  int32_t *c0;
  int8_t *C0;
  int32_t *volatile bias0 = bias_ptr + nc1 * 8;
  if (_nc1 != 0) {
    for (int32_t i = 0; i < mc; i++) {
      C0 = C_ptr + nc1 * 8 + i * ldc;
      c0 = c_ptr + nc1 * 8 + i * NC;
      nc_left = _nc1;
      asm volatile(
          "vdup.32    q15,  %[scale]          \n\t"
          "vdup.8     d24,  %[narrow]         \n\t"
          "cmp        %[_nc1], #4             \n\t"
          "blt        less_four_%=            \n\t"
          "vld1.32    {q0}, [%[c0]]!          \n\t"
          "vld1.32    {q13}, [%[bias0]]!      \n\t"
          "vqadd.s32  q0, q0, q13             \n\t"
          "vcvt.f32.s32 q1, q0                \n\t"
          "vmul.f32   q1, q1, q15             \n\t"
          "vcvt.s32.f32 q2, q1                \n\t"
          "vqmovn.s32 d6, q2                  \n\t"
          "vqmovn.s16 d8, q3                  \n\t"
          "vceq.s8    d9, d8, d24             \n\t"
          "vsub.s8    d8, d8, d9              \n\t"
          "vst1.8     {d8[0]}, [%[C0]]!       \n\t"
          "vst1.8     {d8[1]}, [%[C0]]!       \n\t"
          "vst1.8     {d8[2]}, [%[C0]]!       \n\t"
          "vst1.8     {d8[3]}, [%[C0]]!       \n\t"
          "subs       %[_nc1], %[_nc1], #4    \n\t"
          "beq        process_over_%=         \n\t"
          "less_four_%=:                      \n\t"
          "vld1.32    {q0}, [%[c0]]           \n\t"
          "vld1.32    {q13}, [%[bias0]]       \n\t"
          "vqadd.s32  q0, q0, q13             \n\t"
          "vcvt.f32.s32 q1, q0                \n\t"
          "vmul.f32   q1, q1, q15             \n\t"
          "vcvt.s32.f32 q2, q1                \n\t"
          "vqmovn.s32 d6, q2                  \n\t"
          "vqmovn.s16 d8, q3                  \n\t"
          "vceq.s8    d9, d8, d24             \n\t"
          "vsub.s8    d8, d8, d9              \n\t"
          "loop_save_%=:                      \n\t"
          "vst1.8     {d8[0]}, [%[C0]]!       \n\t"
          "vext.8 d8, d8, d8, #1              \n\t"
          "subs       %[_nc1], %[_nc1], #1    \n\t"
          "bgt        loop_save_%=            \n\t"
          "process_over_%=:                   \n\t"
          :
          : [_nc1] "r"(nc_left), [C0] "r"(C0), [c0] "r"(c0), [bias0] "r"(bias0),
            [scale] "r"(scale), [narrow] "r"(narrow)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q12", "q13", "q15");
    }
  }
#endif  // __aarch64__
#endif  // __ARM_NEON
}

// C = A * B + bias, scale * relu(C), bias is added on column
void Gemm::WriteWithAddReluScale(int32_t mc, int32_t nc, int32_t *c, int8_t *C,
                                 int32_t ldc, int32_t *bias, float scale) {
#if __ARM_NEON
#if __aarch64__
  int32_t nc1 = nc >> 3;
  int32_t _nc1 = nc & 7;

  int32_t *c_ptr;
  int8_t *C_ptr;
  int32x4_t cv0;
  int32x4_t cv1;
  int16x8_t cv_h;
  int8x8_t cv_b;
  int32x4_t biasv;
  int32x4_t zero = vdupq_n_s32(0);
  for (int32_t i = 0; i < mc; ++i) {
    c_ptr = c + i * NC;
    C_ptr = C + i * ldc;
    biasv = vld1q_dup_s32(bias + i);
    for (int32_t j = 0; j < nc1; ++j) {
      cv0 = vld1q_s32(c_ptr);
      cv1 = vld1q_s32(c_ptr + 4);
      cv0 = vqaddq_s32(cv0, biasv);
      cv1 = vqaddq_s32(cv1, biasv);
      cv0 = vmaxq_s32(cv0, zero);
      cv1 = vmaxq_s32(cv1, zero);

      cv_h = vcombine_s16(vqmovn_s32(cv0), vqmovn_s32(cv1));
      cv_b = vqmovn_s16(cv_h);

      vst1_s8(C_ptr, cv_b);
      c_ptr += 8;
      C_ptr += 8;
    }
    if (_nc1 != 0) {
      cv0 = vld1q_s32(c_ptr);
      cv1 = vld1q_s32(c_ptr + 4);
      cv0 = vqaddq_s32(cv0, biasv);
      cv1 = vqaddq_s32(cv1, biasv);
      cv0 = vmaxq_s32(cv0, zero);
      cv1 = vmaxq_s32(cv1, zero);

      cv_h = vcombine_s16(vqmovn_s32(cv0), vqmovn_s32(cv1));
      cv_b = vqmovn_s16(cv_h);

      switch (_nc1) {
        case 7:
          vst1_lane_s8(C_ptr + 6, cv_b, 6);
        case 6:
          vst1_lane_s8(C_ptr + 5, cv_b, 5);
        case 5:
          vst1_lane_s8(C_ptr + 4, cv_b, 4);
        case 4:
          vst1_lane_s8(C_ptr + 3, cv_b, 3);
        case 3:
          vst1_lane_s8(C_ptr + 2, cv_b, 2);
        case 2:
          vst1_lane_s8(C_ptr + 1, cv_b, 1);
        case 1:
          vst1_lane_s8(C_ptr, cv_b, 0);
        default:
          break;
      }
    }
  }
#else
  int32_t zero = 0;
  int32_t nc1 = nc >> 3;
  int32_t _nc1 = nc & 7;
  int32_t step = sizeof(int8_t) * ldc;
  int32_t step1 = sizeof(int32_t) * (NC - (nc1 << 3));
  int32_t volatile m = mc;
  int32_t volatile n = nc1;
  int32_t *volatile c_ptr, *volatile bias_ptr;
  int8_t *volatile C_ptr;
  c_ptr = c;
  C_ptr = C;
  bias_ptr = bias;
  if (nc1 > 0) {
    asm volatile(
        "subs       %[mc], %[mc], #1        \n\t"
        "blt        end_mc_%=               \n\t"
        "vdup.32    q15,  %[scale]          \n\t"
        "vdup.32    q14,  %[zero]           \n\t"
        "loop_mc_%=:                        \n\t"
        "vld1.32    {d26[0]}, [%[bias_ptr]]!\n\t"
        "vdup.32    q13,  d26[0]            \n\t"
        "mov        r6,   %[C_ptr]          \n\t"
        "mov        r5,   %[nc1]            \n\t"
        "subs       r5,   r5,   #1          \n\t"
        "blt        end_nc1_%=              \n\t"
        "loop_nc1_%=:                       \n\t"
        "vld1.32    {q0, q1}, [%[c_ptr]]!   \n\t"
        "vqadd.s32  q0, q0, q13             \n\t"
        "vqadd.s32  q1, q1, q13             \n\t"
        "vmax.s32   q0, q0, q14             \n\t"
        "vmax.s32   q1, q1, q14             \n\t"
        "vcvt.f32.s32 q2, q0                \n\t"
        "vcvt.f32.s32 q3, q1                \n\t"
        "vmul.f32   q2, q2, q15             \n\t"
        "vmul.f32   q3, q3, q15             \n\t"
        "vcvt.s32.f32 q4, q2                \n\t"
        "vcvt.s32.f32 q5, q3                \n\t"
        "vqmovn.s32 d12, q4                 \n\t"
        "vqmovn.s32 d13, q5                 \n\t"
        "vqmovn.s16 d14, q6                 \n\t"
        "vst1.8     {d14}, [r6]!            \n\t"
        "subs       r5,   r5,   #1          \n\t"
        "bge        loop_nc1_%=             \n\t"
        "end_nc1_%=:                        \n\t"

        "add        %[C_ptr], %[C_ptr], %[step]  \n\t"
        "add        %[c_ptr], %[c_ptr], %[step1] \n\t"
        "subs       %[mc], %[mc], #1        \n\t"
        "bge        loop_mc_%=              \n\t"
        "end_mc_%=:                         \n\t"

        :
        : [C_ptr] "r"(C_ptr), [c_ptr] "r"(c_ptr), [mc] "r"(m), [nc1] "r"(n),
          [step] "r"(step), [step1] "r"(step1), [bias_ptr] "r"(bias_ptr),
          [scale] "r"(scale), [zero] "r"(zero)
        : "cc", "memory", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
          "q7", "q13", "q14", "q15");
  }

  int32_t nc_left;
  int32_t *c0;
  int8_t *C0;
  int32_t bias_v;
  if (_nc1 != 0) {
    for (int32_t i = 0; i < mc; i++) {
      C0 = C_ptr + nc1 * 8 + i * ldc;
      c0 = c_ptr + nc1 * 8 + i * NC;
      bias_v = *(bias_ptr + i);
      nc_left = _nc1;
      asm volatile(
          "vdup.32    q15,  %[scale]          \n\t"
          "vdup.32    q14,  %[zero]           \n\t"
          "vdup.32    q13,  %[bias_v]         \n\t"
          "cmp        %[_nc1], #4             \n\t"
          "blt        less_four_%=            \n\t"
          "vld1.32    {q0}, [%[c0]]!          \n\t"
          "vqadd.s32  q0, q0, q13             \n\t"
          "vmax.s32   q0, q0, q14             \n\t"
          "vcvt.f32.s32 q1, q0                \n\t"
          "vmul.f32   q1, q1, q15             \n\t"
          "vcvt.s32.f32 q2, q1                \n\t"
          "vqmovn.s32 d6, q2                  \n\t"
          "vqmovn.s16 d8, q3                  \n\t"
          "vst1.8     {d8[0]}, [%[C0]]!       \n\t"
          "vst1.8     {d8[1]}, [%[C0]]!       \n\t"
          "vst1.8     {d8[2]}, [%[C0]]!       \n\t"
          "vst1.8     {d8[3]}, [%[C0]]!       \n\t"
          "subs       %[_nc1], %[_nc1], #4    \n\t"
          "beq        process_over_%=         \n\t"
          "less_four_%=:                      \n\t"
          "vld1.32    {q0}, [%[c0]]!          \n\t"
          "vqadd.s32  q0, q0, q13             \n\t"
          "vmax.s32   q0, q0, q14             \n\t"
          "vcvt.f32.s32 q1, q0                \n\t"
          "vmul.f32   q1, q1, q15             \n\t"
          "vcvt.s32.f32 q2, q1                \n\t"
          "vqmovn.s32 d6, q2                  \n\t"
          "vqmovn.s16 d8, q3                  \n\t"
          "loop_save_%=:                      \n\t"
          "vst1.8     {d8[0]}, [%[C0]]!       \n\t"
          "vext.8 d8, d8, d8, #1              \n\t"
          "subs       %[_nc1], %[_nc1], #1    \n\t"
          "bgt        loop_save_%=            \n\t"
          "process_over_%=:                   \n\t"
          :
          : [_nc1] "r"(nc_left), [C0] "r"(C0), [c0] "r"(c0),
            [bias_v] "r"(bias_v), [scale] "r"(scale), [zero] "r"(zero)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q13", "q14", "q15");
    }
  }
#endif  // __aarch64__
#endif  // __ARM_NEON
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
