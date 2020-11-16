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

#pragma once

// clang-format off
#define GEMM_DOT_INT8_KERNEL                                           \
  "vld1.s8  {q0}, [%[a_ptr]]!  \n"     /* load a00,a01 to q0, q1*/       \
  "vld1.s8  {d2}, [%[a_ptr]]!  \n"     /* load a00,a01 to q0, q1*/       \
  "veor.s32    q4,  q4, q4     \n"     /* out0 = 0 */                    \
  "veor.s32    q5,  q5, q5     \n"     /* out0 = 0 */                    \
  "veor.s32    q6,  q6, q6     \n"     /* out0 = 0 */                    \
  "veor.s32    q7,  q7, q7     \n"     /* out0 = 0 */                    \
  "veor.s32    q8,  q8, q8     \n"     /* out0 = 0 */                    \
  "veor.s32    q9,  q9, q9     \n"     /* out0 = 0 */                    \
  "veor.s32    q10,  q10, q10  \n"     /* out0 = 0 */                    \
  "veor.s32    q11,  q11, q11  \n"     /* out0 = 0 */                    \
  "veor.s32    q12,  q12, q12  \n"     /* out0 = 0 */                    \
  "veor.s32    q13,  q13, q13  \n"     /* out0 = 0 */                    \
  "veor.s32    q14,  q14, q14  \n"     /* out0 = 0 */                    \
  "veor.s32    q15,  q15, q15  \n"     /* out0 = 0 */                    \
  "cmp   %[k], #0              \n"                                       \
  "beq   2f                    \n"                                       \
  "1:                          \n"                                       \
  "vld1.s8  {q2}, [%[b_ptr]]!  \n"                                       \
  "vld1.s8  {q3}, [%[b_ptr]]!  \n"                                       \
  "vsdot.s8 q4,   q2,  d0[0]   \n"                                       \
  "vsdot.s8 q6,   q2,  d0[1]   \n"                                       \
  "vsdot.s8 q8,   q2,  d1[0]   \n"                                       \
  "vsdot.s8 q10,  q2,  d1[1]   \n"                                       \
  "vsdot.s8 q12,  q2,  d2[0]   \n"                                       \
  "vsdot.s8 q14,  q2,  d2[1]   \n"                                       \
  "vsdot.s8 q5,   q3,  d0[0]   \n"                                       \
  "vsdot.s8 q7,   q3,  d0[1]   \n"                                       \
  "vsdot.s8 q9,   q3,  d1[0]   \n"                                       \
  "vsdot.s8 q11,  q3,  d1[1]   \n"                                       \
  "vsdot.s8 q13,  q3,  d2[0]   \n"                                       \
  "vsdot.s8 q15,  q3,  d2[1]   \n"                                       \
  "vld1.s8  {q0}, [%[a_ptr]]!  \n"     /* load a00,a01 to q0, q1*/       \
  "vld1.s8  {d2}, [%[a_ptr]]!  \n"     /* load a00,a01 to q0, q1*/       \
  "subs %[k], %[k], #1         \n"                                       \
  "bne    1b                   \n"                                       \
  "2:                          \n"
