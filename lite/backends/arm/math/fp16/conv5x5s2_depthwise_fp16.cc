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

#include "lite/backends/arm/math/fp16/common_preprocess.h"
#include "lite/backends/arm/math/fp16/conv_block_utils_fp16.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
// clang-format off
#ifdef __aarch64__
#define FMLA_S2_w0                                           \
  "fmla   v19.8h,  v14.8h,  v0.8h\n" /* outr0 = w0 * r0, 0*/ \
  "fmla   v20.8h,  v14.8h,  v2.8h\n" /* outr1 = w0 * r0, 2*/ \
  "fmla   v21.8h,  v14.8h,  v4.8h\n" /* outr2 = w0 * r0, 4*/ \
  "fmla   v22.8h,  v14.8h,  v6.8h\n" /* outr3 = w0 * r0, 6*/
#define FMLA_S2_w1                                           \
  "fmla   v19.8h,  v15.8h,  v1.8h\n" /* outr0 = w1 * r0, 1*/ \
  "fmla   v20.8h,  v15.8h,  v3.8h\n" /* outr1 = w1 * r0, 3*/ \
  "fmla   v21.8h,  v15.8h,  v5.8h\n" /* outr2 = w1 * r0, 5*/ \
  "fmla   v22.8h,  v15.8h,  v7.8h\n" /* outr3 = w1 * r0, 7*/
#define FMLA_S2_w2                                           \
  "fmla   v19.8h,  v16.8h,  v2.8h\n" /* outr0 = w2 * r0, 2*/ \
  "fmla   v20.8h,  v16.8h,  v4.8h\n" /* outr1 = w2 * r0, 4*/ \
  "fmla   v21.8h,  v16.8h,  v6.8h\n" /* outr2 = w2 * r0, 6*/ \
  "fmla   v22.8h,  v16.8h,  v8.8h\n" /* outr3 = w2 * r0, 8*/
#define FMLA_S2_w3                                           \
  "fmla   v19.8h,  v17.8h,  v3.8h\n" /* outr0 = w3 * r0, 3*/ \
  "fmla   v20.8h,  v17.8h,  v5.8h\n" /* outr1 = w3 * r0, 5*/ \
  "fmla   v21.8h,  v17.8h,  v7.8h\n" /* outr2 = w3 * r0, 7*/ \
  "fmla   v22.8h,  v17.8h,  v9.8h\n" /* outr3 = w3 * r0, 9*/
#define FMLA_S2_w4                                            \
  "fmla   v19.8h,  v18.8h,  v4.8h\n"  /* outr0 = w4 * r0, 4*/ \
  "fmla   v20.8h,  v18.8h,  v6.8h\n"  /* outr1 = w4 * r0, 6*/ \
  "fmla   v21.8h,  v18.8h,  v8.8h\n"  /* outr2 = w4 * r0, 8*/ \
  "fmla   v22.8h,  v18.8h,  v10.8h\n" /* outr3 = w4 * r0, 10*/
#define MULA_S2_R1(ptr0, ptr1)                                            \
  "ldp    q4, q5,   [%[ptr0]], #32\n"                /* load r1, 4-5 */   \
  "ldr    q18,  [%[wc0]], #16\n"                     /* load w4, to q18*/ \
  "ldp    q6, q7,   [%[ptr0]], #32\n"                /* load r0, 6-7 */   \
  FMLA_S2_w0                                                              \
  "ldp    q8, q9,   [%[ptr0]], #32\n" /* load r0, 8-9 */                  \
  FMLA_S2_w1                                                              \
  "ldr    q10,   [%[ptr0]] \n"        /* load r0, 10 */                   \
  FMLA_S2_w2                                                              \
  "sub    %[ptr0], %[ptr0], #32\n"    /* inr1 -= 32 */                    \
  "ldp    q0, q1,   [%[ptr1]], #32\n" /* load r1, 0-1 */                  \
  "ldp    q14, q15, [%[wc0]], #32\n"  /* load w0-1, to q14-15*/           \
  FMLA_S2_w3                                                              \
  "ldp    q16, q17, [%[wc0]], #32\n"  /* load w2-3, to q16-17*/           \
  "ldp    q2, q3,   [%[ptr1]], #32\n" /* load r1, 2-3 */                  \
  FMLA_S2_w4

#define COMPUTE                                                               \
  "mov    x0, %[inr0]\n" \
  "mov    x1, %[inr1]\n" \
  "mov    x2, %[inr2]\n" \
  "mov    x3, %[inr3]\n" \
  "mov    x4, %[inr4]\n" \
  "ldp    q0, q1,   [x0], #32\n" /* load r0, 0-1 */                      \
  "and    v19.16b,  %[vbias].16b, %[vbias].16b\n"                             \
  "ldp    q2, q3,   [x0], #32\n" /* load r0, 2-3 */                      \
  "and    v20.16b,  %[vbias].16b, %[vbias].16b\n"                             \
  "ldp    q4, q5,   [x0], #32\n" /* load r0, 4-5 */                      \
  "and    v21.16b,  %[vbias].16b, %[vbias].16b\n"                             \
  "ldp    q6, q7,   [x0], #32\n" /* load r0, 6-7 */                      \
  "and    v22.16b,  %[vbias].16b, %[vbias].16b\n"                             \
  "ldp    q8, q9,   [x0], #32\n"  /* load r0, 8-9 */                     \
  "fmla   v19.8h,  %[w0].8h,  v0.8h\n" /* outr0 = w0 * r0, 0*/                \
  "fmla   v20.8h,  %[w0].8h,  v2.8h\n" /* outr1 = w0 * r0, 2*/                \
  "fmla   v21.8h,  %[w0].8h,  v4.8h\n" /* outr2 = w0 * r0, 4*/                \
  "fmla   v22.8h,  %[w0].8h,  v6.8h\n" /* outr3 = w0 * r0, 6*/                \
  "ldr    q10,   [x0] \n"         /* load r0, 10 */                      \
  "fmla   v19.8h,  %[w1].8h,  v1.8h\n" /* outr0 = w1 * r0, 1*/                \
  "fmla   v20.8h,  %[w1].8h,  v3.8h\n" /* outr1 = w1 * r0, 3*/                \
  "fmla   v21.8h,  %[w1].8h,  v5.8h\n" /* outr2 = w1 * r0, 5*/                \
  "fmla   v22.8h,  %[w1].8h,  v7.8h\n" /* outr3 = w1 * r0, 7*/                \
  "ldp    q0, q1,   [x1], #32\n"  /* load r1, 0-1 */                     \
  "fmla   v19.8h,  %[w2].8h,  v2.8h\n" /* outr0 = w0 * r0, 2*/                \
  "fmla   v20.8h,  %[w2].8h,  v4.8h\n" /* outr1 = w0 * r0, 4*/                \
  "fmla   v21.8h,  %[w2].8h,  v6.8h\n" /* outr2 = w0 * r0, 6*/                \
  "fmla   v22.8h,  %[w2].8h,  v8.8h\n" /* outr3 = w0 * r0, 8*/                \
  "ldp    q14, q15, [%[wc0]], #32\n"   /* load w0-1, to q14-15*/              \
  "fmla   v19.8h,  %[w3].8h,  v3.8h\n" /* outr0 = w3 * r1, 0*/                \
  "fmla   v20.8h,  %[w3].8h,  v5.8h\n" /* outr1 = w3 * r1, 2*/                \
  "fmla   v21.8h,  %[w3].8h,  v7.8h\n" /* outr2 = w3 * r1, 4*/                \
  "fmla   v22.8h,  %[w3].8h,  v9.8h\n" /* outr3 = w3 * r1, 6*/                \
  "ldp    q16, q17, [%[wc0]], #32\n"   /* load w2-3, to q16-17*/              \
  "ldp    q2, q3,   [x1], #32\n"  /* load r1, 2-3 */                     \
  "fmla   v19.8h,  %[w4].8h,  v4.8h\n" /* outr0 = w3 * r1, 0*/                \
  "fmla   v20.8h,  %[w4].8h,  v6.8h\n" /* outr1 = w3 * r1, 2*/                \
  "fmla   v21.8h,  %[w4].8h,  v8.8h\n" /* outr2 = w3 * r1, 4*/                \
  "fmla   v22.8h,  %[w4].8h,  v10.8h\n" /* outr3 = w3 * r1, 6*/               \
  /* MULA_S2_R1 */                                                            \
  "ldp    q4, q5,   [x1], #32\n"                /* load r1, 4-5 */       \
  "ldr    q18,  [%[wc0]], #16\n"                     /* load w4, to q18*/     \
  "ldp    q6, q7,   [x1], #32\n"                /* load r0, 6-7 */       \
  FMLA_S2_w0                                                                  \
  "ldp    q8, q9,   [x1], #32\n" /* load r0, 8-9 */                      \
  FMLA_S2_w1                                                                  \
  "ldr    q10,   [x1] \n"        /* load r0, 10 */                       \
  FMLA_S2_w2                                                                  \
  "ldp    q0, q1,   [x2], #32\n" /* load r1, 0-1 */                      \
  "ldp    q14, q15, [%[wc0]], #32\n"  /* load w0-1, to q14-15*/               \
  FMLA_S2_w3                                                                  \
  "ldp    q16, q17, [%[wc0]], #32\n"  /* load w2-3, to q16-17*/               \
  "ldp    q2, q3,   [x2], #32\n" /* load r1, 2-3 */                      \
  FMLA_S2_w4                                                                  \
  "ldp    q4, q5,   [x2], #32\n"                /* load r1, 4-5 */       \
  "ldr    q18,  [%[wc0]], #16\n"                     /* load w4, to q18*/     \
  "ldp    q6, q7,   [x2], #32\n"                /* load r0, 6-7 */       \
  FMLA_S2_w0                                                                  \
  "ldp    q8, q9,   [x2], #32\n" /* load r0, 8-9 */                      \
  FMLA_S2_w1                                                                  \
  "ldr    q10,   [x2] \n"        /* load r0, 10 */                       \
  FMLA_S2_w2                                                                  \
  "ldp    q0, q1,   [x3], #32\n" /* load r1, 0-1 */                      \
  "ldp    q14, q15, [%[wc0]], #32\n"  /* load w0-1, to q14-15*/               \
  FMLA_S2_w3                                                                  \
  "ldp    q16, q17, [%[wc0]], #32\n"  /* load w2-3, to q16-17*/               \
  "ldp    q2, q3,   [x3], #32\n" /* load r1, 2-3 */                      \
  FMLA_S2_w4                                                                  \
  "ldp    q4, q5,   [x3], #32\n"                /* load r1, 4-5 */       \
  "ldr    q18,  [%[wc0]], #16\n"                     /* load w4, to q18*/     \
  "ldp    q6, q7,   [x3], #32\n"                /* load r0, 6-7 */       \
  FMLA_S2_w0                                                                  \
  "ldp    q8, q9,   [x3], #32\n" /* load r0, 8-9 */                      \
  FMLA_S2_w1                                                                  \
  "ldr    q10,   [x3] \n"        /* load r0, 10 */                       \
  FMLA_S2_w2                                                                  \
  "ldp    q0, q1,   [x4], #32\n" /* load r1, 0-1 */                      \
  "ldp    q14, q15, [%[wc0]], #32\n"  /* load w0-1, to q14-15*/               \
  FMLA_S2_w3                                                                  \
  "ldp    q16, q17, [%[wc0]], #32\n"  /* load w2-3, to q16-17*/               \
  "ldp    q2, q3,   [x4], #32\n" /* load r1, 2-3 */                      \
  FMLA_S2_w4                                                                  \
  "ldp    q4, q5,   [x4], #32\n"                /* load r1, 4-5 */       \
  "ldr    q18,  [%[wc0]], #16\n"                     /* load w4, to q18*/     \
  "ldp    q6, q7,   [x4], #32\n"                /* load r0, 6-7 */       \
  FMLA_S2_w0                                                                  \
  "ldp    q8, q9,   [x4], #32\n" /* load r0, 8-9 */                      \
  FMLA_S2_w1                                                                  \
  "ldr    q10,   [x4] \n"        /* load r0, 10 */                       \
  FMLA_S2_w2                                                                  \
  FMLA_S2_w3                                                                  \
  FMLA_S2_w4                                                                  \
  "sub    %[wc0], %[wc0], #320\n" /* weight -= 320 */                         \
  "trn1 v0.8h, v19.8h, v20.8h\n"  /* r0: a0a1c0c1*/                           \
  "trn2 v1.8h, v19.8h, v20.8h\n"  /* r0: b0b1d0d1*/                           \
  "trn1 v2.8h, v21.8h, v22.8h\n"  /* r0: a2a3c2c3*/                           \
  "trn2 v3.8h, v21.8h, v22.8h\n"  /* r0: b2b3d2d3*/                           \
  "trn1 v19.4s, v0.4s, v2.4s\n"   /* r0: a0a1a2a3e0e1e2e3*/                   \
  "trn2 v21.4s, v0.4s, v2.4s\n"   /* r0: c0c1c2c3*/                           \
  "trn1 v20.4s, v1.4s, v3.4s\n"   /* r0: b0b1b2b3*/                           \
  "trn2 v22.4s, v1.4s, v3.4s\n"   /* r0: d0d1d2d3*/
#define RELU                      /* relu */     \
  "movi v0.8h, #0\n"              /* for relu */ \
  "fmax v19.8h, v19.8h, v0.8h\n"                 \
  "fmax v20.8h, v20.8h, v0.8h\n"                 \
  "fmax v21.8h, v21.8h, v0.8h\n"                 \
  "fmax v22.8h, v22.8h, v0.8h\n"
#define RELU6 /* relu6 */             \
  "fmin v19.8h, v19.8h, %[vsix].8h\n" \
  "fmin v20.8h, v20.8h, %[vsix].8h\n" \
  "fmin v21.8h, v21.8h, %[vsix].8h\n" \
  "fmin v22.8h, v22.8h, %[vsix].8h\n"
#define LEAKY_RELU                       /* LeakyRelu */ \
  "movi v0.8h, #0\n"                     /* for relu */  \
  "fcmge v1.8h, v19.8h,  v0.8h \n"       /* vcgeq_f32 */ \
  "fmul  v2.8h, v19.8h, %[vscale].8h \n" /* mul */       \
  "fcmge v3.8h, v20.8h,  v0.8h \n"       /* vcgeq_f32 */ \
  "fmul  v4.8h, v20.8h, %[vscale].8h \n" /* mul */       \
  "fcmge v5.8h, v21.8h,  v0.8h \n"       /* vcgeq_f32 */ \
  "fmul  v6.8h, v21.8h, %[vscale].8h \n" /* mul */       \
  "fcmge v7.8h, v22.8h,  v0.8h \n"       /* vcgeq_f32 */ \
  "fmul  v8.8h, v22.8h, %[vscale].8h \n" /* mul */       \
  "bif  v19.16b, v2.16b, v1.16b \n"      /* choose*/     \
  "bif  v20.16b, v4.16b, v3.16b \n"      /* choose*/     \
  "bif  v21.16b, v6.16b, v5.16b \n"      /* choose*/     \
  "bif  v22.16b, v8.16b, v7.16b \n"      /* choose*/
#define STORE                            /* save result */ \
  "trn1 v0.2d, v19.2d, v19.2d\n"         /* r0: a0a1a2a3*/ \
  "trn2 v1.2d, v19.2d, v19.2d\n"         /* r0: e0e1e2e3*/ \
  "trn1 v2.2d, v20.2d, v20.2d\n"         /* r0: b0b1b2b3*/ \
  "trn2 v3.2d, v20.2d, v20.2d\n"         /* r0: f0f1f2f3*/ \
  "trn1 v4.2d, v21.2d, v21.2d\n"         /* r0: c0c1c2c3*/ \
  "trn2 v5.2d, v21.2d, v21.2d\n"         /* r0: g0g1g2g3*/ \
  "trn1 v6.2d, v22.2d, v22.2d\n"         /* r0: d0d1d2d3*/ \
  "trn2 v7.2d, v22.2d, v22.2d\n"         /* r0: h0h1h2h3*/ \
  "st1 {v0.4h}, [%[outc0]], #8\n"                          \
  "st1 {v1.4h}, [%[outc4]], #8\n"                          \
  "st1 {v2.4h}, [%[outc1]], #8\n"                          \
  "st1 {v3.4h}, [%[outc5]], #8\n"                          \
  "st1 {v4.4h}, [%[outc2]], #8\n"                          \
  "st1 {v5.4h}, [%[outc6]], #8\n"                          \
  "st1 {v6.4h}, [%[outc3]], #8\n"                          \
  "st1 {v7.4h}, [%[outc7]], #8\n"

#else

#define COMPUTE                                                              \
  /* fill with bias */                                                       \
  "vld1.16  {d12-d13}, [%[bias]]\n" /* load bias */ /* load weights */       \
  "vld1.16  {d14-d17}, [%[wc0]]!\n"               /* load w0-1, to q7-8 */ \
  "vld1.16  {d0-d3},   [%[inr0]]!\n"                  /* load input r0, 0,1*/  \
  "vand.i16 q12,  q6, q6\n"                                                  \
  "vld1.16  {d4-d7},   [%[inr0]]!\n" /* load input r0, 2,3*/                   \
  "vand.i16 q13,  q6, q6\n"                                                  \
  "vld1.16  {d8-d11},  [%[inr0]]!\n" /* load input r0, 4,5*/                   \
  "vand.i16 q14,  q6, q6\n"                                                  \
  "vand.i16 q15,  q6, q6\n"                                                  \
  "vld1.16  {d12-d13}, [%[inr0]]!\n" /* load input r0, 6*/                     \
  "vmla.f16   q12, q7, q0               @ w0 * inr0\n"                       \
  "vmla.f16   q13, q7, q2               @ w0 * inr2\n"                       \
  "vld1.16    {d18-d21}, [%[wc0]]!\n" /* load w2-3, to q9-q10 */             \
  "vmla.f16   q14, q7, q4               @ w0 * inr4\n"                       \
  "vmla.f16   q15, q7, q6               @ w0 * inr6\n"                       \
  "vmla.f16   q12, q8, q1              @ w1 * inr1\n"                        \
  "vmla.f16   q13, q8, q3              @ w1 * inr3\n"                        \
  "vmla.f16   q14, q8, q5              @ w1 * inr5\n"                        \
  "vld1.16    {d22-d23}, [%[wc0]]!\n" /* load w4, to q11 */                  \
  "vmla.f16   q12, q9, q2              @ w2 * inr2\n"                        \
  "vmla.f16   q13, q9, q4              @ w2 * inr6\n"                        \
  "vmla.f16   q14, q9, q6              @ w2 * inr4\n"                        \
  "vld1.16 {d0-d3}, [%[inr0]]! \n" /* load r0, 7-8 */                          \
  "vmla.f16   q12, q10, q3              @ w3 * inr3\n"                       \
  "vmla.f16   q13, q10, q5              @ w3 * inr5\n"                       \
  "vmla.f16   q14, q10, q0              @ w3 * inr7\n"                       \
  "vmla.f16   q15, q8, q0               @ w1 * inr7\n"                       \
  "vld1.16 {d4-d7}, [%[inr0]] \n" /* load r0, 9-10 */                          \
  "vmla.f16   q12, q11, q4              @ w4 * inr4\n"                       \
  "vmla.f16   q13, q11, q6              @ w4 * inr6\n"                       \
  "vmla.f16   q14, q11, q1              @ w4 * inr8\n"                       \
  "vmla.f16   q15, q9, q1               @ w2 * inr8\n"                       \
  "vld1.16    {d0-d3}, [%[inr1]]!         @ load r1, 0, 1\n"                   \
  "vld1.16    {d14-d17}, [%[wc0]]!\n" /* load w0-1, to q7-8 */               \
  "vmla.f16   q15, q10, q2               @ w3 * inr9\n"                      \
  "vld1.16    {d4-d5}, [%[inr1]]!         @ load r1, 2\n"                      \
  "sub %[inr0], %[inr0], #16         @ r0 - 16 to nextline address\n"        \
  "vld1.16    {d18-d21}, [%[wc0]]!\n" /* load w2-3, to q9-10 */              \
  "vmla.f16   q12, q7, q0              @ w0 * inr0\n"                        \
  "vmla.f16   q13, q7, q2              @ w0 * inr2\n"                        \
  "vmla.f16   q15, q11, q3               @ w4 * inr10\n"                     \
  "vld1.16    {d6-d9}, [%[inr1]]!         @ load r1, 3, 4\n"                   \
  "vld1.16    {d22-d23}, [%[wc0]]!\n" /* load w4, to q11 */                  \
  "vld1.16    {d10-d13}, [%[inr1]]!       @ load r1, 5, 6\n"                   \
  "vmla.f16   q14, q7, q4              @ w0 * inr0\n"                        \
  "vmla.f16   q15, q7, q6              @ w0 * inr2\n"                        \
  "vmla.f16   q12, q8, q1              @ w1 * inr1\n"                        \
  "vmla.f16   q13, q8, q3              @ w1 * inr3\n"                        \
  "vld1.16    {d0-d3}, [%[inr1]]!         @ load r1, 7, 8\n"                   \
  "vmla.f16   q14, q8, q5              @ w1 * inr5\n"                        \
  "vmla.f16   q15, q8, q0              @ w1 * inr7\n"                        \
  "vmla.f16   q12, q9, q2              @ w2 * inr2\n"                        \
  "vmla.f16   q13, q9, q4              @ w2 * inr4\n"                        \
  "vmla.f16   q14, q9, q6              @ w2 * inr6\n"                        \
  "vmla.f16   q15, q9, q1              @ w2 * inr8\n"                        \
  "vmla.f16   q12, q10, q3              @ w3 * inr3\n"                       \
  "vld1.16    {d4-d7}, [%[inr1]]         @ load r1, 9, 10\n"                   \
  "vmla.f16   q13, q10, q5              @ w3 * inr5\n"                       \
  "vmla.f16   q14, q10, q0              @ w3 * inr7\n"                       \
  "vmla.f16   q15, q10, q2              @ w3 * inr9\n"                       \
  "vld1.16    {d14-d17}, [%[wc0]]!\n" /* load w0-1, to q7-8 */               \
  "vmla.f16   q12, q11, q4              @ w4 * inr4\n"                       \
  "vmla.f16   q13, q11, q6              @ w4 * inr6\n"                       \
  "vmla.f16   q14, q11, q1              @ w4 * inr8\n"                       \
  "vmla.f16   q15, q11, q3              @ w4 * inr10\n"                      \
  "vld1.16    {d0-d3}, [%[inr2]]!         @ load r2, 0, 1\n"                   \
  "vld1.16    {d18-d21}, [%[wc0]]!\n" /* load w2-3, to q9-10 */              \
  "sub %[inr1], %[inr1], #16           @ r1 - 16 to nextline address\n"     \
  "vld1.16    {d4-d7}, [%[inr2]]!         @ load r2, 2, 3\n"                   \
  "vld1.16    {d22-d23}, [%[wc0]]!\n" /* load w4 to q11 */                   \
  "vmla.f16   q12, q7, q0              @ w0 * inr0\n"                        \
  "vmla.f16   q13, q7, q2              @ w0 * inr2\n"                        \
  "vld1.16    {d8-d11}, [%[inr2]]!         @ load r2, 4, 5\n"                  \
  "vmla.f16   q12, q8, q1              @ w1 * inr1\n"                        \
  "vmla.f16   q13, q8, q3              @ w1 * inr3\n"                        \
  "vld1.16    {d12-d13}, [%[inr2]]!         @ load r2, 6 \n"                   \
  "vmla.f16   q14, q7, q4              @ w0 * inr4\n"                        \
  "vmla.f16   q15, q7, q6              @ w0 * inr6\n"                        \
  "vld1.16    {d0-d3}, [%[inr2]]!         @ load r2, 7, 8\n"                   \
  "vmla.f16   q12, q9, q2              @ w2 * inr2\n"                        \
  "vmla.f16   q13, q9, q4              @ w2 * inr4\n"                        \
  "vmla.f16   q14, q8, q5              @ w1 * inr5\n"                        \
  "vmla.f16   q15, q8, q0              @ w1 * inr7\n"                        \
  "vmla.f16   q12, q10, q3              @ w3 * inr3\n"                       \
  "vmla.f16   q13, q10, q5              @ w3 * inr5\n"                       \
  "vmla.f16   q14, q9, q6              @ w2 * inr6\n"                        \
  "vmla.f16   q15, q9, q1              @ w2 * inr8\n"                        \
  "vld1.16    {d4-d7}, [%[inr2]]         @ load r2, 9, 10\n"                   \
  "vmla.f16   q12, q11, q4              @ w4 * inr4\n"                       \
  "vmla.f16   q13, q11, q6              @ w4 * inr6\n"                       \
  "vmla.f16   q14, q10, q0              @ w3 * inr7\n"                       \
  "vmla.f16   q15, q10, q2              @ w3 * inr9\n"                       \
  "vld1.16    {d14-d17}, [%[wc0]]!\n" /* load w0-1, to q7-8 */               \
  "sub %[inr2], %[inr2], #16            @ r1 - 16 to nextline address\n"     \
  "vmla.f16   q14, q11, q1              @ w4 * inr8\n"                       \
  "vld1.16    {d0-d3}, [%[inr3]]!         @ load r3, 0, 1\n"                   \
  "vmla.f16   q15, q11, q3              @ w4 * inr10\n"                      \
  "vld1.16    {d4-d7}, [%[inr3]]!         @ load r3, 2, 3\n"                   \
  "vld1.16    {d18-d21}, [%[wc0]]!\n" /* load w2-3, to q9-10 */              \
  "vmla.f16   q12, q7, q0              @ w0 * inr0\n"                        \
  "vmla.f16   q13, q7, q2              @ w0 * inr2\n"                        \
  "vld1.16    {d8-d11}, [%[inr3]]!         @ load r3, 4, 5\n"                  \
  "vld1.16    {d22-d23}, [%[wc0]]!\n" /* load w4 to q11 */                   \
  "vld1.16    {d12-d13}, [%[inr3]]!         @ load r3, 6, \n"                  \
  "vmla.f16   q12, q8, q1              @ w1 * inr1\n"                        \
  "vmla.f16   q13, q8, q3              @ w1 * inr3\n"                        \
  "vmla.f16   q14, q7, q4              @ w0 * inr4\n"                        \
  "vmla.f16   q15, q7, q6              @ w0 * inr6\n"                        \
  "vld1.16    {d0-d3}, [%[inr3]]!         @ load r3, 7, 8\n"                   \
  "vmla.f16   q12, q9, q2              @ w2 * inr2\n"                        \
  "vmla.f16   q13, q9, q4              @ w2 * inr4\n"                        \
  "vmla.f16   q14, q8, q5              @ w1 * inr5\n"                        \
  "vmla.f16   q15, q8, q0              @ w1 * inr7\n"                        \
  "vmla.f16   q12, q10, q3              @ w3 * inr3\n"                       \
  "vld1.16    {d4-d7}, [%[inr3]]         @ load r3, 9, 10\n"                   \
  "vmla.f16   q13, q10, q5              @ w3 * inr5\n"                       \
  "vmla.f16   q14, q9, q6              @ w2 * inr6\n"                        \
  "vmla.f16   q15, q9, q1              @ w2 * inr8\n"                        \
  "vmla.f16   q12, q11, q4              @ w4 * inr4\n"                       \
  "vmla.f16   q13, q11, q6              @ w4 * inr6\n"                       \
  "vmla.f16   q14, q10, q0              @ w3 * inr7\n"                       \
  "vmla.f16   q15, q10, q2              @ w3 * inr9\n"                       \
  "vld1.16    {d14-d17}, [%[wc0]]!\n" /* load w0-1, to q7-8 */               \
  "sub %[inr3], %[inr3], #16           @ r1 - 16 to nextline address\n"     \
  "vmla.f16   q14, q11, q1              @ w4 * inr8\n"                       \
  "vld1.16    {d0-d3}, [%[inr4]]!         @ load r4, 0, 1\n"                   \
  "vmla.f16   q15, q11, q3              @ w4 * inr10\n"                      \
  "vld1.16    {d4-d7}, [%[inr4]]!         @ load r4, 2, 3\n"                   \
  "vld1.16    {d18-d21}, [%[wc0]]!\n" /* load w2-3, to q9-10 */              \
  "vmla.f16   q12, q7, q0              @ w0 * inr0\n"                        \
  "vmla.f16   q13, q7, q2              @ w0 * inr2\n"                        \
  "vld1.16    {d8-d11}, [%[inr4]]!         @ load r3, 4, 5\n"                  \
  "vld1.16    {d22-d23}, [%[wc0]]!\n" /* load w4 to q11 */                   \
  "vld1.16    {d12-d13}, [%[inr4]]!         @ load r3, 6, \n"                  \
  "vmla.f16   q12, q8, q1              @ w1 * inr1\n"                        \
  "vmla.f16   q13, q8, q3              @ w1 * inr3\n"                        \
  "vmla.f16   q14, q7, q4              @ w0 * inr4\n"                        \
  "vmla.f16   q15, q7, q6              @ w0 * inr6\n"                        \
  "vld1.16    {d0-d3}, [%[inr4]]!         @ load r3, 7, 8\n"                   \
  "vmla.f16   q12, q9, q2              @ w2 * inr2\n"                        \
  "vmla.f16   q13, q9, q4              @ w2 * inr4\n"                        \
  "vmla.f16   q14, q8, q5              @ w1 * inr5\n"                        \
  "vmla.f16   q15, q8, q0              @ w1 * inr7\n"                        \
  "vmla.f16   q12, q10, q3              @ w3 * inr3\n"                       \
  "vld1.16    {d4-d7}, [%[inr4]]         @ load r3, 9, 10\n"                   \
  "vmla.f16   q13, q10, q5              @ w3 * inr5\n"                       \
  "vmla.f16   q14, q9, q6              @ w2 * inr6\n"                        \
  "vmla.f16   q15, q9, q1              @ w2 * inr8\n"                        \
  "vmla.f16   q12, q11, q4              @ w4 * inr4\n"                       \
  "vmla.f16   q13, q11, q6              @ w4 * inr6\n"                       \
  "vmla.f16   q14, q10, q0              @ w3 * inr7\n"                       \
  "vmla.f16   q15, q10, q2              @ w3 * inr9\n"                       \
  "sub    %[wc0], %[wc0], #400          @ wc0 - 400 to start address\n"      \
  "sub %[inr4], %[inr4], #16            @ r1 - 16 to nextline address\n"     \
  "vmla.f16   q14, q11, q1              @ w4 * inr8\n"                       \
  "vmla.f16   q15, q11, q3              @ w4 * inr10\n"                      \

#define RELU /* relu */             \
  "vmov.u16 q0, #0\n"               \
  "vld1.16 {d2-d3}, [%[six_ptr]]\n" \
  "vmax.f16 q12, q12, q0\n"         \
  "vmax.f16 q13, q13, q0\n"         \
  "vmax.f16 q14, q14, q0\n"         \
  "vmax.f16 q15, q15, q0\n"
#define RELU6 /* relu6 */   \
  "vmin.f16 q12, q12, q1\n" \
  "vmin.f16 q13, q13, q1\n" \
  "vmin.f16 q14, q14, q1\n" \
  "vmin.f16 q15, q15, q1\n"
#define LEAKY_RELU /* LeakyRelu */    \
  "vmov.u16 q0, #0\n"                 \
  "vld1.16 {d2-d3}, [%[scale_ptr]]\n" \
  "vcge.f16 q2, q12, q0  @ q0 > 0 \n" \
  "vcge.f16 q4, q13, q0  @ q0 > 0 \n" \
  "vcge.f16 q6, q14, q0  @ q0 > 0 \n" \
  "vcge.f16 q8, q15, q0  @ q0 > 0 \n" \
  "vmul.f16 q3, q12, q1   @ mul \n"   \
  "vmul.f16 q5, q13, q1   @ mul \n"   \
  "vmul.f16 q7, q14, q1   @ mul \n"   \
  "vmul.f16 q9, q15, q1   @ mul \n"   \
  "vbif q12, q3, q2 @ choose \n"      \
  "vbif q13, q5, q4 @ choose \n"      \
  "vbif q14, q7, q6 @ choose \n"      \
  "vbif q15, q9, q8 @ choose \n"
#define STORE                        /* save result */ \
  "vst1.16 {d24-d25}, [%[outc0]]\n" /* save outc0*/   \
  "vst1.16 {d26-d27}, [%[outc1]]\n" /* save outc1*/   \
  "vst1.16 {d28-d29}, [%[outc2]]\n" /* save outc2*/   \
  "vst1.16 {d30-d31}, [%[outc3]]\n" /* save outc3*/


#endif
// clang-format on

void act_switch_5x5s2(const float16_t* inr0,
                      const float16_t* inr1,
                      const float16_t* inr2,
                      const float16_t* inr3,
                      const float16_t* inr4,
                      float16_t* outc0,
                      float16_t* outc1,
                      float16_t* outc2,
                      float16_t* outc3,
                      float16_t* outc4,
                      float16_t* outc5,
                      float16_t* outc6,
                      float16_t* outc7,
                      float16x8_t w0,
                      float16x8_t w1,
                      float16x8_t w2,
                      float16x8_t w3,
                      float16x8_t w4,
                      float16x8_t vbias,
                      const float16_t* weight_c,
                      float16_t* bias_local,
                      const operators::ActivationParam act_param) {
  bool has_active = act_param.has_active;
  if (has_active) {
    float16_t tmp = act_param.Relu_clipped_coef;
    float16_t ss = act_param.Leaky_relu_alpha;
#ifdef __aarch64__
    float16x8_t vsix = vdupq_n_f16(tmp);
    float16x8_t vscale = vdupq_n_f16(ss);
#else
    float16x8_t vsix[8] = {tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp};
    float16x8_t vscale[8] = {ss, ss, ss, ss, ss, ss, ss, ss};
#endif
    switch (act_param.active_type) {
      case lite_api::ActivationType::kRelu:
#ifdef __aarch64__
        asm volatile(COMPUTE RELU STORE
                     : [wc0] "+r"(weight_c),
                       [outc0] "+r"(outc0),
                       [outc1] "+r"(outc1),
                       [outc2] "+r"(outc2),
                       [outc3] "+r"(outc3),
                       [outc4] "+r"(outc4),
                       [outc5] "+r"(outc5),
                       [outc6] "+r"(outc6),
                       [outc7] "+r"(outc7)
                     : [inr0] "r"(inr0),
                       [inr1] "r"(inr1),
                       [inr2] "r"(inr2),
                       [inr3] "r"(inr3),
                       [inr4] "r"(inr4),
                       [w0] "w"(w0),
                       [w1] "w"(w1),
                       [w2] "w"(w2),
                       [w3] "w"(w3),
                       [w4] "w"(w4),
                       [vbias] "w"(vbias)
                     : "cc",
                       "memory",
                       "x0",
                       "x1",
                       "x2",
                       "x3",
                       "x4",
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
                       "v14",
                       "v15",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20",
                       "v21",
                       "v22");
        inr0 += 64;
        inr1 += 64;
        inr2 += 64;
        inr3 += 64;
        inr4 += 64;
#else
        asm volatile(COMPUTE RELU STORE
                     : [inr0] "+r"(inr0),
                       [inr1] "+r"(inr1),
                       [inr2] "+r"(inr2),
                       [inr3] "+r"(inr3),
                       [inr4] "+r"(inr4),
                       [wc0] "+r"(weight_c),
                       [outc0] "+r"(outc0),
                       [outc1] "+r"(outc1),
                       [outc2] "+r"(outc2),
                       [outc3] "+r"(outc3)
                     : [bias] "r"(bias_local), [six_ptr] "r"(vsix)
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
#endif
        break;
      case lite_api::ActivationType::kRelu6:
#ifdef __aarch64__
        asm volatile(COMPUTE RELU RELU6 STORE
                     : [wc0] "+r"(weight_c),
                       [outc0] "+r"(outc0),
                       [outc1] "+r"(outc1),
                       [outc2] "+r"(outc2),
                       [outc3] "+r"(outc3),
                       [outc4] "+r"(outc4),
                       [outc5] "+r"(outc5),
                       [outc6] "+r"(outc6),
                       [outc7] "+r"(outc7)
                     : [inr0] "r"(inr0),
                       [inr1] "r"(inr1),
                       [inr2] "r"(inr2),
                       [inr3] "r"(inr3),
                       [inr4] "r"(inr4),
                       [w0] "w"(w0),
                       [w1] "w"(w1),
                       [w2] "w"(w2),
                       [w3] "w"(w3),
                       [w4] "w"(w4),
                       [vbias] "w"(vbias),
                       [vsix] "w"(vsix)
                     : "cc",
                       "memory",
                       "x0",
                       "x1",
                       "x2",
                       "x3",
                       "x4",
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
                       "v14",
                       "v15",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20",
                       "v21",
                       "v22");
        inr0 += 64;
        inr1 += 64;
        inr2 += 64;
        inr3 += 64;
        inr4 += 64;
#else
        asm volatile(COMPUTE RELU RELU6 STORE
                     : [inr0] "+r"(inr0),
                       [inr1] "+r"(inr1),
                       [inr2] "+r"(inr2),
                       [inr3] "+r"(inr3),
                       [inr4] "+r"(inr4),
                       [wc0] "+r"(weight_c),
                       [outc0] "+r"(outc0),
                       [outc1] "+r"(outc1),
                       [outc2] "+r"(outc2),
                       [outc3] "+r"(outc3)
                     : [bias] "r"(bias_local), [six_ptr] "r"(vsix)
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
#endif
        break;
      case lite_api::ActivationType::kLeakyRelu:
#ifdef __aarch64__
        asm volatile(COMPUTE LEAKY_RELU STORE
                     : [wc0] "+r"(weight_c),
                       [outc0] "+r"(outc0),
                       [outc1] "+r"(outc1),
                       [outc2] "+r"(outc2),
                       [outc3] "+r"(outc3),
                       [outc4] "+r"(outc4),
                       [outc5] "+r"(outc5),
                       [outc6] "+r"(outc6),
                       [outc7] "+r"(outc7)
                     : [inr0] "r"(inr0),
                       [inr1] "r"(inr1),
                       [inr2] "r"(inr2),
                       [inr3] "r"(inr3),
                       [inr4] "r"(inr4),
                       [w0] "w"(w0),
                       [w1] "w"(w1),
                       [w2] "w"(w2),
                       [w3] "w"(w3),
                       [w4] "w"(w4),
                       [vbias] "w"(vbias),
                       [vscale] "w"(vscale)
                     : "cc",
                       "memory",
                       "x0",
                       "x1",
                       "x2",
                       "x3",
                       "x4",
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
                       "v14",
                       "v15",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20",
                       "v21",
                       "v22");
        inr0 += 64;
        inr1 += 64;
        inr2 += 64;
        inr3 += 64;
        inr4 += 64;
#else
        asm volatile(COMPUTE LEAKY_RELU STORE
                     : [inr0] "+r"(inr0),
                       [inr1] "+r"(inr1),
                       [inr2] "+r"(inr2),
                       [inr3] "+r"(inr3),
                       [inr4] "+r"(inr4),
                       [wc0] "+r"(weight_c),
                       [outc0] "+r"(outc0),
                       [outc1] "+r"(outc1),
                       [outc2] "+r"(outc2),
                       [outc3] "+r"(outc3)
                     : [bias] "r"(bias_local), [scale_ptr] "r"(vscale)
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
#endif
        break;
      default:
        LOG(FATAL) << "this act_type: "
                   << static_cast<int>(act_param.active_type)
                   << " fuse not support";
    }
  } else {
#ifdef __aarch64__
    asm volatile(COMPUTE STORE
                 : [wc0] "+r"(weight_c),
                   [outc0] "+r"(outc0),
                   [outc1] "+r"(outc1),
                   [outc2] "+r"(outc2),
                   [outc3] "+r"(outc3),
                   [outc4] "+r"(outc4),
                   [outc5] "+r"(outc5),
                   [outc6] "+r"(outc6),
                   [outc7] "+r"(outc7)
                 : [inr0] "r"(inr0),
                   [inr1] "r"(inr1),
                   [inr2] "r"(inr2),
                   [inr3] "r"(inr3),
                   [inr4] "r"(inr4),
                   [w0] "w"(w0),
                   [w1] "w"(w1),
                   [w2] "w"(w2),
                   [w3] "w"(w3),
                   [w4] "w"(w4),
                   [vbias] "w"(vbias)
                 : "cc",
                   "memory",
                   "x0",
                   "x1",
                   "x2",
                   "x3",
                   "x4",
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
                   "v14",
                   "v15",
                   "v16",
                   "v17",
                   "v18",
                   "v19",
                   "v20",
                   "v21",
                   "v22");
    inr0 += 64;
    inr1 += 64;
    inr2 += 64;
    inr3 += 64;
    inr4 += 64;
#else
    asm volatile(COMPUTE STORE
                 : [inr0] "+r"(inr0),
                   [inr1] "+r"(inr1),
                   [inr2] "+r"(inr2),
                   [inr3] "+r"(inr3),
                   [inr4] "+r"(inr4),
                   [wc0] "+r"(weight_c),
                   [outc0] "+r"(outc0),
                   [outc1] "+r"(outc1),
                   [outc2] "+r"(outc2),
                   [outc3] "+r"(outc3)
                 : [bias] "r"(bias_local)
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
#endif
  }
}

void conv_depthwise_5x5s2_fp16(const float16_t* i_data,
                               float16_t* o_data,
                               int bs,
                               int oc,
                               int oh,
                               int ow,
                               int ic,
                               int ih,
                               int win,
                               const float16_t* weights,
                               const float16_t* bias,
                               const operators::ConvParam& param,
                               ARMContext* ctx) {
  auto paddings = *param.paddings;
  int threads = ctx->threads();
  const int pad_h = paddings[0];
  const int pad_w = paddings[2];
  const int out_c_block = 8;
  const int out_h_kernel = 1;
  const int out_w_kernel = 4;
  const int win_ext = ow * 2 + 3;
  const int ow_round = ROUNDUP(ow, out_w_kernel);
  const int win_round = ROUNDUP(win_ext, out_w_kernel);
  const int hin_round = oh * 2 + 3;
  const int prein_size = win_round * hin_round * out_c_block;
  auto workspace_size = threads * prein_size + win_round + ow_round;
  ctx->ExtendWorkspace(sizeof(float16_t) * workspace_size);

  bool flag_bias = param.bias != nullptr;
  auto act_param = param.activation_param;

  /// get workspace
  auto ptr_zero = ctx->workspace_data<float16_t>();
  memset(ptr_zero, 0, sizeof(float16_t) * win_round);
  float16_t* ptr_write = ptr_zero + win_round;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;

  int ws = -pad_w;
  int we = ws + win_round;
  int hs = -pad_h;
  int he = hs + hin_round;
  int w_loop = ow_round / out_w_kernel;
  auto remain = w_loop * out_w_kernel - ow;
  bool flag_remain = remain > 0;
  remain = out_w_kernel - remain;
  remain = remain > 0 ? remain : 0;
  int row_len = win_round * out_c_block;

  float16x8_t vzero = vdupq_n_f16(0.f);

  for (int n = 0; n < bs; ++n) {
    const float16_t* din_batch = i_data + n * ic * size_in_channel;
    float16_t* dout_batch = o_data + n * oc * size_out_channel;
    LITE_PARALLEL_COMMON_BEGIN(c, tid, oc, 0, out_c_block) {
#ifdef LITE_USE_THREAD_POOL
      float16_t* pre_din = ptr_write + ow_round + tid * prein_size;
#elif ARM_WITH_OMP
      float16_t* pre_din =
          ptr_write + ow_round + omp_get_thread_num() * prein_size;
#else
      float16_t* pre_din = ptr_write + ow_round;
#endif
      /// const array size
      prepack_input_nxwc8_fp16_dw(
          din_batch, pre_din, c, hs, he, ws, we, ic, win, ih, ptr_zero);
      const float16_t* weight_c = weights + c * 25;  // kernel_w * kernel_h
      float16_t* dout_c00 = dout_batch + c * size_out_channel;
      float16_t bias_local[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#ifdef __aarch64__
      float16x8_t w0 = vld1q_f16(weight_c);       // w0, v23
      float16x8_t w1 = vld1q_f16(weight_c + 8);   // w1, v24
      float16x8_t w2 = vld1q_f16(weight_c + 16);  // w2, v25
      float16x8_t w3 = vld1q_f16(weight_c + 24);  // w3, v26
      float16x8_t w4 = vld1q_f16(weight_c + 32);  // w4, v27
      float16x8_t vbias = vdupq_n_f16(0.f);
      weight_c += 40;
      if (flag_bias) {
        if (c + out_c_block < oc) {
          vbias = vld1q_f16(&bias[c]);  // v28
        } else {
          int k = 0;
          for (; k < 8 && c + k < oc; k++) {
            bias_local[k] = bias[c + k];
          }
          vbias = vld1q_f16(bias_local);  // v28
        }
      }
#else
      if (flag_bias) {
        for (int k = 0; k < 8 && c + k < oc; k++) {
          bias_local[k] = bias[c + k];
        }
      }
#endif
      for (int h = 0; h < oh; h += out_h_kernel) {
        float16_t* outc0 = dout_c00 + h * ow;
        float16_t* outc1 = outc0 + size_out_channel;
        float16_t* outc2 = outc1 + size_out_channel;
        float16_t* outc3 = outc2 + size_out_channel;
        float16_t* outc4 = outc3 + size_out_channel;
        float16_t* outc5 = outc4 + size_out_channel;
        float16_t* outc6 = outc5 + size_out_channel;
        float16_t* outc7 = outc6 + size_out_channel;
        const float16_t* inr0 = pre_din + h * 2 * row_len;
        const float16_t* inr1 = inr0 + row_len;
        const float16_t* inr2 = inr1 + row_len;
        const float16_t* inr3 = inr2 + row_len;
        const float16_t* inr4 = inr3 + row_len;

        if (c + out_c_block > oc) {
          switch (c + out_c_block - oc) {
            case 7:
              outc1 = ptr_write;
            case 6:
              outc2 = ptr_write;
            case 5:
              outc3 = ptr_write;
            case 4:
              outc4 = ptr_write;
            case 3:
              outc5 = ptr_write;
            case 2:
              outc6 = ptr_write;
            case 1:
              outc7 = ptr_write;
            default:
              break;
          }
        }
        auto c0 = outc0;
        auto c1 = outc1;
        auto c2 = outc2;
        auto c3 = outc3;
        auto c4 = outc4;
        auto c5 = outc5;
        auto c6 = outc6;
        auto c7 = outc7;
        float16_t pre_out[64];
        for (int w = 0; w < w_loop; ++w) {
          bool flag_mask = (w == w_loop - 1) && flag_remain;
          if (flag_mask) {
            c0 = outc0;
            c1 = outc1;
            c2 = outc2;
            c3 = outc3;
            c4 = outc4;
            c5 = outc5;
            c6 = outc6;
            c7 = outc7;
            outc0 = pre_out;
            outc1 = pre_out + 8;
            outc2 = pre_out + 16;
            outc3 = pre_out + 24;
            outc4 = pre_out + 32;
            outc5 = pre_out + 40;
            outc6 = pre_out + 48;
            outc7 = pre_out + 56;
          }
#ifdef __aarch64__
          act_switch_5x5s2(inr0,
                           inr1,
                           inr2,
                           inr3,
                           inr4,
                           outc0,
                           outc1,
                           outc2,
                           outc3,
                           outc4,
                           outc5,
                           outc6,
                           outc7,
                           w0,
                           w1,
                           w2,
                           w3,
                           w4,
                           vbias,
                           weight_c,
                           bias_local,
                           act_param);
#else
          float16_t pre_out_[32];
          float16_t *pre_din0_ = &(pre_out_[0]), *pre_din1_ = &(pre_out_[8]),
                    *pre_din2_ = &(pre_out_[16]), *pre_din3_ = &(pre_out_[24]);
          act_switch_5x5s2(inr0,
                           inr1,
                           inr2,
                           inr3,
                           inr4,
                           &pre_out_[0],
                           &pre_out_[8],
                           &pre_out_[16],
                           &pre_out_[24],
                           &pre_out[32],
                           &pre_out[40],
                           &pre_out[48],
                           &pre_out[56],
                           vzero,
                           vzero,
                           vzero,
                           vzero,
                           vzero,
                           vzero,
                           weight_c,
                           bias_local,
                           act_param);
          asm volatile(
              "vld1.32 {d0-d1},  [%[r0]]\n"
              "vld1.32 {d2-d3},  [%[r1]]\n"
              "vld1.32 {d4-d5},  [%[r2]]\n"
              "vld1.32 {d6-d7},  [%[r3]]\n"
              "vtrn.16   q0, q1\n"
              "vtrn.16   q2, q3\n"
              "vtrn.32   q0, q2\n"
              "vtrn.32   q1, q3\n"

              "vswp      d1, d2\n"
              "vswp      d5, d6\n"
              "vst1.16 {d0}, [%[outc0]]\n"
              "vst1.16 {d1}, [%[outc1]]\n"
              "vst1.16 {d4}, [%[outc2]]\n"
              "vst1.16 {d5}, [%[outc3]]\n"
              "vst1.16 {d2}, [%[outc4]]\n"
              "vst1.16 {d3}, [%[outc5]]\n"
              "vst1.16 {d6}, [%[outc6]]\n"
              "vst1.16 {d7}, [%[outc7]]\n"

              : [r0] "+r"(pre_din0_),
                [r1] "+r"(pre_din1_),
                [r2] "+r"(pre_din2_),
                [r3] "+r"(pre_din3_),
                [outc0] "+r"(outc0),
                [outc1] "+r"(outc1),
                [outc2] "+r"(outc2),
                [outc3] "+r"(outc3),
                [outc4] "+r"(outc4),
                [outc5] "+r"(outc5),
                [outc6] "+r"(outc6),
                [outc7] "+r"(outc7)
              :
              : "cc", "memory", "q0", "q1", "q2", "q3");

#endif
          if (flag_mask) {
            for (int i = 0; i < remain; ++i) {
              c0[i] = pre_out[i];
              c1[i] = pre_out[i + 8];
              c2[i] = pre_out[i + 16];
              c3[i] = pre_out[i + 24];
              c4[i] = pre_out[i + 32];
              c5[i] = pre_out[i + 40];
              c6[i] = pre_out[i + 48];
              c7[i] = pre_out[i + 56];
            }
          }
          inr0 += 64;
          inr1 += 64;
          inr2 += 64;
          inr3 += 64;
          inr4 += 64;
          outc0 += 4;
          outc1 += 4;
          outc2 += 4;
          outc3 += 4;
          outc4 += 4;
          outc5 += 4;
          outc6 += 4;
          outc7 += 4;
        }
      }
    }
    LITE_PARALLEL_COMMON_END();
  }
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
