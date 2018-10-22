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
#include "memory/t_malloc.h"
#include "operators/math/gemm.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

// 8 bits int small block inner product
void Gemm::AddDot6x8(int32_t k, const int8_t *a, const int8_t *b, int32_t *c,
                     int32_t ldc) {
#if __ARM_NEON
#if __aarch64__
// TODO
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
      "vmov.s8      q4,         #0                 \n\t"
      "vmov.s8      q5,         #0                 \n\t"
      "vmov.s8      q6,         #0                 \n\t"
      "vmov.s8      q7,         #0                 \n\t"
      "vmov.s8      q8,         #0                 \n\t"
      "vmov.s8      q9,         #0                 \n\t"
      "vmov.s8      q10,        #0                 \n\t"
      "vmov.s8      q11,        #0                 \n\t"
      "vmov.s8      q12,        #0                 \n\t"
      "vmov.s8      q13,        #0                 \n\t"
      "vmov.s8      q14,        #0                 \n\t"
      "vmov.s8      q15,        #0                 \n\t"
      "mov r0,      #12                            \n\t"
      "subs         %[kc1],     %[kc1], #1         \n\t"
      "blt          1f                             \n\t"
      "0:                                          \n\t"
      "pld          [%[a_ptr], #64]                \n\t"
      "pld          [%[b_ptr], #128]               \n\t"
      "vld1.s8      {d0-d2},    [%[a_ptr]]!        \n\t"  // A 4 cols, q0 used,
                                                          // 1/2 q3 used
      "vmov.s8      q2,         #0                 \n\t"  // q2 used
      "vld1.s8      {d6-d7},    [%[b_ptr]]!        \n\t"  // B 2 rows, B row1,
                                                          // q1
      "vdup.s8      d3,         d0[0]              \n\t"  // q3 used // used
      "vmlal.s8     q2,         d6,            d3  \n\t"  // A col00 * B row0
      "vdup.s8      d3,         d0[6]              \n\t"  // q3 used
      "vmlal.s8     q2,         d7,            d3  \n\t"  // A col10 * B row1,
                                                          // q3 free
      "vaddw.s16    q4,         q4,            d4  \n\t"
      "vaddw.s16    q5,         q5,            d5  \n\t"  // res row 0
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d0[1]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d0[7]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q6,         q6,            d4  \n\t"
      "vaddw.s16    q7,         q7,            d5  \n\t"  // res row 1
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d0[2]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[0]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q8,         q8,            d4  \n\t"
      "vaddw.s16    q9,         q9,            d5  \n\t"  // res row 2
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d0[3]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[1]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q10,        q10,           d4  \n\t"
      "vaddw.s16    q11,        q11,           d5  \n\t"  // res row 3
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d0[4]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[2]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q12,        q12,           d4  \n\t"
      "vaddw.s16    q13,        q13,           d5  \n\t"  // res row 4
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d0[5]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[3]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q14,        q14,           d4  \n\t"
      "vaddw.s16    q15,        q15,           d5  \n\t"  // res row 5

      "vld1.s8      {d6-d7},    [%[b_ptr]]!        \n\t"  // B 2 rows, B row1,
                                                          // q1
      "vmov.s8      q2,         #0                 \n\t"  // q2 used
      "vdup.s8      d3,         d1[4]              \n\t"  // q3 used // used
      "vmlal.s8     q2,         d6,            d3  \n\t"  // A col00 * B row0
      "vdup.s8      d3,         d2[2]              \n\t"  // q3 used
      "vmlal.s8     q2,         d7,            d3  \n\t"  // A col10 * B row1,
                                                          // q3 free
      "vaddw.s16    q4,         q4,            d4  \n\t"
      "vaddw.s16    q5,         q5,            d5  \n\t"  // res row 0
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d1[5]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[3]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q6,         q6,            d4  \n\t"
      "vaddw.s16    q7,         q7,            d5  \n\t"  // res row 1
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d1[6]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[4]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q8,         q8,            d4  \n\t"
      "vaddw.s16    q9,         q9,            d5  \n\t"  // res row 2
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d1[7]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[5]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q10,        q10,           d4  \n\t"
      "vaddw.s16    q11,        q11,           d5  \n\t"  // res row 3
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d2[0]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[6]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q12,        q12,           d4  \n\t"
      "vaddw.s16    q13,        q13,           d5  \n\t"  // res row 4
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d2[1]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[7]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q14,        q14,           d4  \n\t"
      "vaddw.s16    q15,        q15,           d5  \n\t"  // res row 5

      "vld1.s8      {d0-d2},    [%[a_ptr]]!        \n\t"  // A 4 cols, q0 used,
                                                          // 1/2 q3 used
      "vmov.s8      q2,         #0                 \n\t"  // q2 used
      "vld1.s8      {d6-d7},    [%[b_ptr]]!        \n\t"  // B 2 rows, B row1,
                                                          // q1
      "vdup.s8      d3,         d0[0]              \n\t"  // q3 used // used
      "vmlal.s8     q2,         d6,            d3  \n\t"  // A col00 * B row0
      "vdup.s8      d3,         d0[6]              \n\t"  // q3 used
      "vmlal.s8     q2,         d7,            d3  \n\t"  // A col10 * B row1,
                                                          // q3 free
      "vaddw.s16    q4,         q4,            d4  \n\t"
      "vaddw.s16    q5,         q5,            d5  \n\t"  // res row 0
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d0[1]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d0[7]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q6,         q6,            d4  \n\t"
      "vaddw.s16    q7,         q7,            d5  \n\t"  // res row 1
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d0[2]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[0]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q8,         q8,            d4  \n\t"
      "vaddw.s16    q9,         q9,            d5  \n\t"  // res row 2
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d0[3]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[1]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q10,        q10,           d4  \n\t"
      "vaddw.s16    q11,        q11,           d5  \n\t"  // res row 3
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d0[4]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[2]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q12,        q12,           d4  \n\t"
      "vaddw.s16    q13,        q13,           d5  \n\t"  // res row 4
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d0[5]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[3]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q14,        q14,           d4  \n\t"
      "vaddw.s16    q15,        q15,           d5  \n\t"  // res row 5

      "vld1.s8      {d6-d7},    [%[b_ptr]]!        \n\t"  // B 2 rows, B row1,
                                                          // q1
      "vmov.s8      q2,         #0                 \n\t"  // q2 used
      "vdup.s8      d3,         d1[4]              \n\t"  // q3 used // used
      "vmlal.s8     q2,         d6,            d3  \n\t"  // A col00 * B row0
      "vdup.s8      d3,         d2[2]              \n\t"  // q3 used
      "vmlal.s8     q2,         d7,            d3  \n\t"  // A col10 * B row1,
                                                          // q3 free
      "vaddw.s16    q4,         q4,            d4  \n\t"
      "vaddw.s16    q5,         q5,            d5  \n\t"  // res row 0
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d1[5]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[3]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q6,         q6,            d4  \n\t"
      "vaddw.s16    q7,         q7,            d5  \n\t"  // res row 1
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d1[6]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[4]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q8,         q8,            d4  \n\t"
      "vaddw.s16    q9,         q9,            d5  \n\t"  // res row 2
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d1[7]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[5]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q10,        q10,           d4  \n\t"
      "vaddw.s16    q11,        q11,           d5  \n\t"  // res row 3
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d2[0]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[6]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q12,        q12,           d4  \n\t"
      "vaddw.s16    q13,        q13,           d5  \n\t"  // res row 4
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d2[1]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[7]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q14,        q14,           d4  \n\t"
      "vaddw.s16    q15,        q15,           d5  \n\t"  // res row 5

      "subs         %[kc1],     %[kc1],        #1  \n\t"
      "bge          0b                             \n\t"
      "1:                                          \n\t"  // last <8 rows
      "subs         %[kc3],     %[kc3],        #1  \n\t"
      "blt          2f                             \n\t"
      "vld1.s8      {d0-d2},    [%[a_ptr]]!        \n\t"
      "vmov.s8      q2,         #0                 \n\t"
      "vld1.s8      {d6-d7},    [%[b_ptr]]!        \n\t"
      "vdup.s8      d3,         d0[0]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d0[6]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q4,         q4,            d4  \n\t"
      "vaddw.s16    q5,         q5,            d5  \n\t"  // res row 0
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d0[1]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d0[7]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q6,         q6,            d4  \n\t"
      "vaddw.s16    q7,         q7,            d5  \n\t"  // res row 1
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d0[2]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[0]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q8,         q8,            d4  \n\t"
      "vaddw.s16    q9,         q9,            d5  \n\t"  // res row 2
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d0[3]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[1]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q10,        q10,           d4  \n\t"
      "vaddw.s16    q11,        q11,           d5  \n\t"  // res row 3
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d0[4]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[2]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q12,        q12,           d4  \n\t"
      "vaddw.s16    q13,        q13,           d5  \n\t"  // res row 4
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d0[5]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d1[3]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q14,        q14,           d4  \n\t"
      "vaddw.s16    q15,        q15,           d5  \n\t"  // res row 5

      "vld1.s8      {d6-d7},    [%[b_ptr]]!        \n\t"
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d1[4]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[2]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q4,         q4,            d4  \n\t"
      "vaddw.s16    q5,         q5,            d5  \n\t"  // res row 0
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d1[5]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[3]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q6,         q6,            d4  \n\t"
      "vaddw.s16    q7,         q7,            d5  \n\t"  // res row 1
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d1[6]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[4]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q8,         q8,            d4  \n\t"
      "vaddw.s16    q9,         q9,            d5  \n\t"  // res row 2
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d3,         d1[7]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[5]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q10,        q10,           d4  \n\t"
      "vaddw.s16    q11,        q11,           d5  \n\t"  // res row 3
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d2[0]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[6]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q12,        q12,           d4  \n\t"
      "vaddw.s16    q13,        q13,           d5  \n\t"  // res row 4
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d3,         d2[1]              \n\t"
      "vmlal.s8     q2,         d6,            d3  \n\t"
      "vdup.s8      d3,         d2[7]              \n\t"
      "vmlal.s8     q2,         d7,            d3  \n\t"
      "vaddw.s16    q14,        q14,           d4  \n\t"
      "vaddw.s16    q15,        q15,           d5  \n\t"  // res row 5

      "2:                                          \n\t"  // last <4 rows
      "subs         %[kc5],     %[kc5],        #1  \n\t"
      "blt          3f                             \n\t"
      "vld1.s8      {d0, d1},   [%[a_ptr]],    r0  \n\t"
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d6,         d0[0]              \n\t"
      "vld1.s8      {d2-d3},    [%[b_ptr]]!        \n\t"
      "vdup.s8      d7,         d0[6]              \n\t"
      "vmlal.s8     q2,         d2,            d6  \n\t"
      "vmlal.s8     q2,         d3,            d7  \n\t"
      "vaddw.s16    q4,         q4,            d4  \n\t"
      "vaddw.s16    q5,         q5,            d5  \n\t"  // res row 0
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d6,         d0[1]              \n\t"
      "vdup.s8      d7,         d0[7]              \n\t"
      "vmlal.s8     q2,         d2,            d6  \n\t"
      "vmlal.s8     q2,         d3,            d7  \n\t"
      "vaddw.s16    q6,         q6,            d4  \n\t"
      "vaddw.s16    q7,         q7,            d5  \n\t"  // res row 1
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d6,         d0[2]              \n\t"
      "vdup.s8      d7,         d1[0]              \n\t"
      "vmlal.s8     q2,         d2,            d6  \n\t"
      "vmlal.s8     q2,         d3,            d7  \n\t"
      "vaddw.s16    q8,         q8,            d4  \n\t"
      "vaddw.s16    q9,         q9,            d5  \n\t"  // res row 2
      "vmov.s8      q2,                        #0  \n\t"
      "vdup.s8      d6,         d0[3]              \n\t"
      "vdup.s8      d7,         d1[1]              \n\t"
      "vmlal.s8     q2,         d2,            d6  \n\t"
      "vmlal.s8     q2,         d3,            d7  \n\t"
      "vaddw.s16    q10,        q10,           d4  \n\t"
      "vaddw.s16    q11,        q11,           d5  \n\t"  // res row 3
      "vmov.s8      q2,         #0.                \n\t"
      "vdup.s8      d6,         d0[4]              \n\t"
      "vdup.s8      d7,         d1[2]              \n\t"
      "vmlal.s8     q2,         d2,            d6  \n\t"
      "vmlal.s8     q2,         d3,            d7  \n\t"
      "vaddw.s16    q12,        q12,           d4  \n\t"
      "vaddw.s16    q13,        q13,           d5  \n\t"  // res row 4
      "vmov.s8      q2,         #0                 \n\t"
      "vdup.s8      d6,         d0[5]              \n\t"
      "vdup.s8      d7,         d1[3]              \n\t"
      "vmlal.s8     q2,         d2,            d6  \n\t"
      "vmlal.s8     q2,         d3,            d7  \n\t"
      "vaddw.s16    q14,        q14,           d4  \n\t"
      "vaddw.s16    q15,        q15,           d5  \n\t"  // res row 5

      "3:                                          \n\t"  // last <2 rows
      "subs         %[kc6],     %[kc6],        #1  \n\t"
      "blt          4f                             \n\t"
      "vld1.s8      {d0},       [%[a_ptr]]         \n\t"
      "vld1.s8      {d1},       [%[b_ptr]]         \n\t"
      "vdup.s8      d2,         d0[0]              \n\t"
      "vmull.s8     q2,         d1,            d2  \n\t"
      "vaddw.s16    q4,         q4,            d4  \n\t"
      "vaddw.s16    q5,         q5,            d5  \n\t"  // res row 0
      "vdup.s8      d2,         d0[1]              \n\t"
      "vmull.s8     q2,         d1,            d2  \n\t"
      "vaddw.s16    q6,         q6,            d4  \n\t"
      "vaddw.s16    q7,         q7,            d5  \n\t"  // res row 1
      "vdup.s8      d2,         d0[2]              \n\t"
      "vmull.s8     q2,         d1,            d2  \n\t"
      "vaddw.s16    q8,         q8,            d4  \n\t"
      "vaddw.s16    q9,         q9,            d5  \n\t"  // res row 2
      "vdup.s8      d2,         d0[3]              \n\t"
      "vmull.s8     q2,         d1,            d2  \n\t"
      "vaddw.s16    q10,        q10,           d4  \n\t"
      "vaddw.s16    q11,        q11,           d5  \n\t"  // res row 3
      "vdup.s8      d2,         d0[4]              \n\t"
      "vmull.s8     q2,         d1,            d2  \n\t"
      "vaddw.s16    q12,        q12,           d4  \n\t"
      "vaddw.s16    q13,        q13,           d5  \n\t"  // res row 4
      "vdup.s8      d2,         d0[5]              \n\t"
      "vmull.s8     q2,         d1,            d2  \n\t"
      "vaddw.s16    q14,        q14,           d4  \n\t"
      "vaddw.s16    q15,        q15,           d5  \n\t"  // res row 4
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
void Gemm::InnerKernelWithBias(int32_t mc, int32_t nc, int8_t alpha,
                               const int8_t *a, const int8_t *b, int8_t beta,
                               int32_t *c, int32_t *C, int32_t ldc, bool relu,
                               int8_t *bias) {
#pragma omp parallel for
  for (int32_t j = 0; j < nc; j += NR) {
    for (int32_t i = 0; i < mc; i += MR) {
      AddDot6x8(KC, a + i * KC, b + j * KC, c + i * NC + j, NC);
    }
  }
  if (alpha != 1) {
    WriteWithAlphaBeta(mc, nc, c, C, ldc);
    return;
  }
  if (beta == 0) {
    WriteBasic(mc, nc, c, C, ldc);
    return;
  }
  if (beta == 1 && !relu) {
    if (bias == nullptr) {
      WriteWithAdd(mc, nc, c, C, ldc);
    } else {
      WriteWithAddV1(mc, nc, c, C, ldc, bias);
    }
    return;
  }
  if (beta == 1 && relu) {
    if (bias == nullptr) {
      WriteWithAddRelu(mc, nc, c, C, ldc);
    } else {
      WriteWithAddReluV1(mc, nc, c, C, ldc, bias);
    }
    return;
  }
}

// 8 bits int PackMatrixA
void Gemm::PackMatrixA_6r(int32_t m, int32_t k, int32_t m_tail, const int8_t *A,
                          int32_t lda, int8_t *buffer) {
  const int32_t i_length = m - m_tail;
  for (int32_t i = 0; i < i_length; i += MR) {
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
  for (int32_t j = 0; j < j_length; j += NR) {
    int8_t *local_buffer = buffer + j * k;
    for (int32_t i = 0; i < k; ++i) {
      const int8_t *b0 = &B(i, j);
#if __ARM_NEON
#if __aarch64__
      // TODO
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
      for (int32_t j = n; j < j_length + NR; ++j) {
        *local_buffer++ = 0;
      }
    }
  }
}

// 8 bits int matrix product (m*k x k*n)
void Gemm::Sgemm(int32_t m, int32_t n, int32_t k, int8_t alpha, const int8_t *A,
                 int32_t lda, const int8_t *B, int32_t ldb, int8_t beta,
                 int32_t *C, int32_t ldc, bool relu, int8_t *bias) {
  // L1 data cache is 32 kib (Per Contex-A57, Contex-A72, Contex-A73)
  // L2 cache is 0.5~4 Mib (Contex-A72 cluster)
  int32_t L1 = 32 * 1024;
  int32_t L2 = 512 * 1024;

  KC = k;
  MC = L1 / (KC * sizeof(int8_t));
  NC = L2 / (KC * sizeof(int8_t));

  // make sure MC is multiple of MR, and NC is multiple of NR
  if (MC == 0) {
    MC = MR;
  } else {
    int32_t mblock_num = (m + MC - 1) / MC;
    MC = (m + mblock_num - 1) / mblock_num;
    MC = (MC + MR - 1) / MR * MR;
  }
  // DLOG << "mblock_num = " << mblock_num << ", MC = " << MC << "\n";
  if (NC == 0) {
    NC = NR;
  } else {
    int32_t nblock_num = (n + NC - 1) / NC;
    NC = (n + nblock_num - 1) / nblock_num;
    NC = (NC + NR - 1) / NR * NR;
  }
  //  DLOG << "nblock_num = " << nblock_num << ", NC = " << NC << "\n";
  packedA_int8 = static_cast<int8_t *>(
      paddle_mobile::memory::Alloc(sizeof(int8_t) * MC * KC));
  packedB_int8 = static_cast<int8_t *>(
      paddle_mobile::memory::Alloc(sizeof(int8_t) * KC * NC));
  packedC_int8 = static_cast<int32_t *>(
      paddle_mobile::memory::Alloc(sizeof(int32_t) * MC * NC));
  zero_int8 =
      static_cast<int8_t *>(paddle_mobile::memory::Alloc(sizeof(int8_t) * KC));

  memset(static_cast<void *>(zero_int8), 0, sizeof(int8_t) * KC);
  int32_t mc, nc;
  for (int32_t j = 0; j < n; j += NC) {
    nc = s_min(n - j, NC);
    PackMatrixB_8c(KC, nc, nc % NR, &B(0, j), ldb, packedB_int8);
    for (int32_t i = 0; i < m; i += MC) {
      mc = s_min(m - i, MC);
      PackMatrixA_6r(mc, KC, mc % MR, &A(i, 0), lda, packedA_int8);
      if (bias == nullptr) {
        InnerKernelWithBias(mc, nc, alpha, packedA_int8, packedB_int8, beta,
                            packedC_int8, &C(i, j), ldc, relu, nullptr);
      } else {
        InnerKernelWithBias(mc, nc, alpha, packedA_int8, packedB_int8, beta,
                            packedC_int8, &C(i, j), ldc, relu, bias + i);
      }
    }
  }

  paddle_mobile::memory::Free(packedA_int8);
  paddle_mobile::memory::Free(packedB_int8);
  paddle_mobile::memory::Free(packedC_int8);
  paddle_mobile::memory::Free(zero_int8);
}

//  8 bits int write back
// C = alpha * A * B + beta * C
void Gemm::WriteWithAlphaBeta(int32_t mc, int32_t nc, int32_t *c, int32_t *C,
                              int32_t ldc) {}
// C = A * B, 8位 int32_t
void Gemm::WriteBasic(int32_t mc, int32_t nc, int32_t *c, int32_t *C,
                      int32_t ldc) {
#if __ARM_NEON
#if __aarch64__
// TODO
#else
  int32_t nc1 = nc >> 4;
  int32_t _nc1 = nc & 15;
  int32_t step = sizeof(int32_t) * ldc;
  int32_t step1 = sizeof(int32_t) * (NC - (nc1 << 4));
  int32_t volatile m = mc;

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
        : [C_ptr] "r"(C_ptr), [c_ptr] "r"(c_ptr), [mc] "r"(m), [nc1] "r"(nc1),
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

// C = A * B + C
void Gemm::WriteWithAdd(int32_t mc, int32_t nc, int32_t *c, int32_t *C,
                        int32_t ldc) {}

// C = A * B + bias
void Gemm::WriteWithAddV1(int32_t mc, int32_t nc, int32_t *c, int32_t *C,
                          int32_t ldc, int8_t *bias) {}
// C = A * B + C, relu(C)
void Gemm::WriteWithAddRelu(int32_t mc, int32_t nc, int32_t *c, int32_t *C,
                            int32_t ldc) {}

// C = A * B + bias, relu(C)
void Gemm::WriteWithAddReluV1(int32_t mc, int32_t nc, int32_t *c, int32_t *C,
                              int32_t ldc, int8_t *bias) {}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
