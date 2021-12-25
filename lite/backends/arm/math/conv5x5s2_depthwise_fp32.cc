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

#include <arm_neon.h>
#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_depthwise.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"
#include "lite/operators/op_params.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {
#ifdef __aarch64__
#define COMPUTE                                                    \
  "ldp    q0, q1,   [%[inr0]], #32\n" /* load r0, 0-1 */           \
  "and    v19.16b,  %[vbias].16b, %[vbias].16b\n"                  \
  "ldp    q2, q3,   [%[inr0]], #32\n" /* load r0, 2-3 */           \
  "and    v20.16b,  %[vbias].16b, %[vbias].16b\n"                  \
  "ldp    q4, q5,   [%[inr0]], #32\n" /* load r0, 4-5 */           \
  "and    v21.16b,  %[vbias].16b, %[vbias].16b\n"                  \
  "ldp    q6, q7,   [%[inr0]], #32\n" /* load r0, 6-7 */           \
  "and    v22.16b,  %[vbias].16b, %[vbias].16b\n"                  \
  "ldp    q8, q9,   [%[inr0]], #32\n"    /* load r0, 8-9 */        \
  "fmla   v19.4s ,  %[w0].4s,  v0.4s\n"  /* outr0 = w0 * r0, 0*/   \
  "fmla   v20.4s ,  %[w0].4s,  v2.4s\n"  /* outr1 = w0 * r0, 2*/   \
  "fmla   v21.4s ,  %[w0].4s,  v4.4s\n"  /* outr2 = w0 * r0, 4*/   \
  "fmla   v22.4s ,  %[w0].4s,  v6.4s\n"  /* outr3 = w0 * r0, 6*/   \
  "ldr    q10,   [%[inr0]] \n"           /* load r0, 10 */         \
  "fmla   v19.4s ,  %[w1].4s,  v1.4s\n"  /* outr0 = w1 * r0, 1*/   \
  "fmla   v20.4s ,  %[w1].4s,  v3.4s\n"  /* outr1 = w1 * r0, 3*/   \
  "fmla   v21.4s ,  %[w1].4s,  v5.4s\n"  /* outr2 = w1 * r0, 5*/   \
  "fmla   v22.4s ,  %[w1].4s,  v7.4s\n"  /* outr3 = w1 * r0, 7*/   \
  "sub    %[inr0], %[inr0], #32\n"       /* inr0 -= 32 */          \
  "ldp    q0, q1,   [%[inr1]], #32\n"    /* load r1, 0-1 */        \
  "fmla   v19.4s ,  %[w2].4s,  v2.4s\n"  /* outr0 = w0 * r0, 2*/   \
  "fmla   v20.4s ,  %[w2].4s,  v4.4s\n"  /* outr1 = w0 * r0, 4*/   \
  "fmla   v21.4s ,  %[w2].4s,  v6.4s\n"  /* outr2 = w0 * r0, 6*/   \
  "fmla   v22.4s ,  %[w2].4s,  v8.4s\n"  /* outr3 = w0 * r0, 8*/   \
  "ldp    q14, q15, [%[wc0]], #32\n"     /* load w0-1, to q14-15*/ \
  "fmla   v19.4s ,  %[w3].4s,  v3.4s\n"  /* outr0 = w3 * r1, 0*/   \
  "fmla   v20.4s ,  %[w3].4s,  v5.4s\n"  /* outr1 = w3 * r1, 2*/   \
  "fmla   v21.4s ,  %[w3].4s,  v7.4s\n"  /* outr2 = w3 * r1, 4*/   \
  "fmla   v22.4s ,  %[w3].4s,  v9.4s\n"  /* outr3 = w3 * r1, 6*/   \
  "ldp    q16, q17, [%[wc0]], #32\n"     /* load w2-3, to q16-17*/ \
  "ldp    q2, q3,   [%[inr1]], #32\n"    /* load r1, 2-3 */        \
  "fmla   v19.4s ,  %[w4].4s,  v4.4s\n"  /* outr0 = w3 * r1, 0*/   \
  "fmla   v20.4s ,  %[w4].4s,  v6.4s\n"  /* outr1 = w3 * r1, 2*/   \
  "fmla   v21.4s ,  %[w4].4s,  v8.4s\n"  /* outr2 = w3 * r1, 4*/   \
  "fmla   v22.4s ,  %[w4].4s,  v10.4s\n" /* outr3 = w3 * r1, 6*/   \
  "ldp    q4, q5,   [%[inr1]], #32\n"    /* load r1, 4-5 */        \
  "ldr    q18,  [%[wc0]], #16\n"         /* load w4, to q18*/      \
  "ldp    q6, q7,   [%[inr1]], #32\n"    /* load r0, 6-7 */        \
  "fmla   v19.4s ,  v14.4s,  v0.4s\n"    /* outr0 = w0 * r0, 0*/   \
  "fmla   v20.4s ,  v14.4s,  v2.4s\n"    /* outr1 = w0 * r0, 2*/   \
  "fmla   v21.4s ,  v14.4s,  v4.4s\n"    /* outr2 = w0 * r0, 4*/   \
  "fmla   v22.4s ,  v14.4s,  v6.4s\n"    /* outr3 = w0 * r0, 6*/   \
  "ldp    q8, q9,   [%[inr1]], #32\n"    /* load r0, 8-9 */        \
  "fmla   v19.4s ,  v15.4s,  v1.4s\n"    /* outr0 = w1 * r0, 1*/   \
  "fmla   v20.4s ,  v15.4s,  v3.4s\n"    /* outr1 = w1 * r0, 3*/   \
  "fmla   v21.4s ,  v15.4s,  v5.4s\n"    /* outr2 = w1 * r0, 5*/   \
  "fmla   v22.4s ,  v15.4s,  v7.4s\n"    /* outr3 = w1 * r0, 7*/   \
  "ldr    q10,   [%[inr1]] \n"           /* load r0, 10 */         \
  "fmla   v19.4s ,  v16.4s,  v2.4s\n"    /* outr0 = w0 * r0, 2*/   \
  "fmla   v20.4s ,  v16.4s,  v4.4s\n"    /* outr1 = w0 * r0, 4*/   \
  "fmla   v21.4s ,  v16.4s,  v6.4s\n"    /* outr2 = w0 * r0, 6*/   \
  "fmla   v22.4s ,  v16.4s,  v8.4s\n"    /* outr3 = w0 * r0, 8*/   \
  "sub    %[inr1], %[inr1], #32\n"       /* inr1 -= 32 */          \
  "ldp    q0, q1,   [%[inr2]], #32\n"    /* load r1, 0-1 */        \
  "ldp    q14, q15, [%[wc0]], #32\n"     /* load w0-1, to q14-15*/ \
  "fmla   v19.4s ,  v17.4s,  v3.4s\n"    /* outr0 = w3 * r1, 0*/   \
  "fmla   v20.4s ,  v17.4s,  v5.4s\n"    /* outr1 = w3 * r1, 2*/   \
  "fmla   v21.4s ,  v17.4s,  v7.4s\n"    /* outr2 = w3 * r1, 4*/   \
  "fmla   v22.4s ,  v17.4s,  v9.4s\n"    /* outr3 = w3 * r1, 6*/   \
  "ldp    q16, q17, [%[wc0]], #32\n"     /* load w2-3, to q16-17*/ \
  "ldp    q2, q3,   [%[inr2]], #32\n"    /* load r1, 2-3 */        \
  "fmla   v19.4s ,  v18.4s,  v4.4s\n"    /* outr0 = w3 * r1, 0*/   \
  "fmla   v20.4s ,  v18.4s,  v6.4s\n"    /* outr1 = w3 * r1, 2*/   \
  "fmla   v21.4s ,  v18.4s,  v8.4s\n"    /* outr2 = w3 * r1, 4*/   \
  "fmla   v22.4s ,  v18.4s,  v10.4s\n"   /* outr3 = w3 * r1, 6*/   \
  "ldp    q4, q5,   [%[inr2]], #32\n"    /* load r1, 4-5 */        \
  "ldr    q18,  [%[wc0]], #16\n"         /* load w4, to q18*/      \
  "ldp    q6, q7,   [%[inr2]], #32\n"    /* load r0, 6-7 */        \
  "fmla   v19.4s ,  v14.4s,  v0.4s\n"    /* outr0 = w0 * r0, 0*/   \
  "fmla   v20.4s ,  v14.4s,  v2.4s\n"    /* outr1 = w0 * r0, 2*/   \
  "fmla   v21.4s ,  v14.4s,  v4.4s\n"    /* outr2 = w0 * r0, 4*/   \
  "fmla   v22.4s ,  v14.4s,  v6.4s\n"    /* outr3 = w0 * r0, 6*/   \
  "ldp    q8, q9,   [%[inr2]], #32\n"    /* load r0, 8-9 */        \
  "fmla   v19.4s ,  v15.4s,  v1.4s\n"    /* outr0 = w1 * r0, 1*/   \
  "fmla   v20.4s ,  v15.4s,  v3.4s\n"    /* outr1 = w1 * r0, 3*/   \
  "fmla   v21.4s ,  v15.4s,  v5.4s\n"    /* outr2 = w1 * r0, 5*/   \
  "fmla   v22.4s ,  v15.4s,  v7.4s\n"    /* outr3 = w1 * r0, 7*/   \
  "ldr    q10,   [%[inr2]] \n"           /* load r0, 10 */         \
  "fmla   v19.4s ,  v16.4s,  v2.4s\n"    /* outr0 = w0 * r0, 2*/   \
  "fmla   v20.4s ,  v16.4s,  v4.4s\n"    /* outr1 = w0 * r0, 4*/   \
  "fmla   v21.4s ,  v16.4s,  v6.4s\n"    /* outr2 = w0 * r0, 6*/   \
  "fmla   v22.4s ,  v16.4s,  v8.4s\n"    /* outr3 = w0 * r0, 8*/   \
  "sub    %[inr2], %[inr2], #32\n"       /* inr0 -= 32 */          \
  "ldp    q0, q1,   [%[inr3]], #32\n"    /* load r1, 0-1 */        \
  "ldp    q14, q15, [%[wc0]], #32\n"     /* load w0-1, to q14-15*/ \
  "fmla   v19.4s ,  v17.4s,  v3.4s\n"    /* outr0 = w3 * r1, 0*/   \
  "fmla   v20.4s ,  v17.4s,  v5.4s\n"    /* outr1 = w3 * r1, 2*/   \
  "fmla   v21.4s ,  v17.4s,  v7.4s\n"    /* outr2 = w3 * r1, 4*/   \
  "fmla   v22.4s ,  v17.4s,  v9.4s\n"    /* outr3 = w3 * r1, 6*/   \
  "ldp    q16, q17, [%[wc0]], #32\n"     /* load w2-3, to q16-17*/ \
  "ldp    q2, q3,   [%[inr3]], #32\n"    /* load r1, 2-3 */        \
  "fmla   v19.4s ,  v18.4s,  v4.4s\n"    /* outr0 = w3 * r1, 0*/   \
  "fmla   v20.4s ,  v18.4s,  v6.4s\n"    /* outr1 = w3 * r1, 2*/   \
  "fmla   v21.4s ,  v18.4s,  v8.4s\n"    /* outr2 = w3 * r1, 4*/   \
  "fmla   v22.4s ,  v18.4s,  v10.4s\n"   /* outr3 = w3 * r1, 6*/   \
  "ldp    q4, q5,   [%[inr3]], #32\n"    /* load r1, 4-5 */        \
  "ldr    q18,  [%[wc0]], #16\n"         /* load w4, to q18*/      \
  "ldp    q6, q7,   [%[inr3]], #32\n"    /* load r0, 6-7 */        \
  "fmla   v19.4s ,  v14.4s,  v0.4s\n"    /* outr0 = w0 * r0, 0*/   \
  "fmla   v20.4s ,  v14.4s,  v2.4s\n"    /* outr1 = w0 * r0, 2*/   \
  "fmla   v21.4s ,  v14.4s,  v4.4s\n"    /* outr2 = w0 * r0, 4*/   \
  "fmla   v22.4s ,  v14.4s,  v6.4s\n"    /* outr3 = w0 * r0, 6*/   \
  "ldp    q8, q9,   [%[inr3]], #32\n"    /* load r0, 8-9 */        \
  "fmla   v19.4s ,  v15.4s,  v1.4s\n"    /* outr0 = w1 * r0, 1*/   \
  "fmla   v20.4s ,  v15.4s,  v3.4s\n"    /* outr1 = w1 * r0, 3*/   \
  "fmla   v21.4s ,  v15.4s,  v5.4s\n"    /* outr2 = w1 * r0, 5*/   \
  "fmla   v22.4s ,  v15.4s,  v7.4s\n"    /* outr3 = w1 * r0, 7*/   \
  "ldr    q10,   [%[inr3]] \n"           /* load r0, 10 */         \
  "fmla   v19.4s ,  v16.4s,  v2.4s\n"    /* outr0 = w0 * r0, 2*/   \
  "fmla   v20.4s ,  v16.4s,  v4.4s\n"    /* outr1 = w0 * r0, 4*/   \
  "fmla   v21.4s ,  v16.4s,  v6.4s\n"    /* outr2 = w0 * r0, 6*/   \
  "fmla   v22.4s ,  v16.4s,  v8.4s\n"    /* outr3 = w0 * r0, 8*/   \
  "sub    %[inr3], %[inr3], #32\n"       /* inr0 -= 32 */          \
  "ldp    q0, q1,   [%[inr4]], #32\n"    /* load r1, 0-1 */        \
  "ldp    q14, q15, [%[wc0]], #32\n"     /* load w0-1, to q14-15*/ \
  "fmla   v19.4s ,  v17.4s,  v3.4s\n"    /* outr0 = w3 * r1, 0*/   \
  "fmla   v20.4s ,  v17.4s,  v5.4s\n"    /* outr1 = w3 * r1, 2*/   \
  "fmla   v21.4s ,  v17.4s,  v7.4s\n"    /* outr2 = w3 * r1, 4*/   \
  "fmla   v22.4s ,  v17.4s,  v9.4s\n"    /* outr3 = w3 * r1, 6*/   \
  "ldp    q16, q17, [%[wc0]], #32\n"     /* load w2-3, to q16-17*/ \
  "ldp    q2, q3,   [%[inr4]], #32\n"    /* load r1, 2-3 */        \
  "fmla   v19.4s ,  v18.4s,  v4.4s\n"    /* outr0 = w3 * r1, 0*/   \
  "fmla   v20.4s ,  v18.4s,  v6.4s\n"    /* outr1 = w3 * r1, 2*/   \
  "fmla   v21.4s ,  v18.4s,  v8.4s\n"    /* outr2 = w3 * r1, 4*/   \
  "fmla   v22.4s ,  v18.4s,  v10.4s\n"   /* outr3 = w3 * r1, 6*/   \
  "ldp    q4, q5,   [%[inr4]], #32\n"    /* load r1, 4-5 */        \
  "ldr    q18,  [%[wc0]], #16\n"         /* load w4, to q18*/      \
  "ldp    q6, q7,   [%[inr4]], #32\n"    /* load r0, 6-7 */        \
  "fmla   v19.4s ,  v14.4s,  v0.4s\n"    /* outr0 = w0 * r0, 0*/   \
  "fmla   v20.4s ,  v14.4s,  v2.4s\n"    /* outr1 = w0 * r0, 2*/   \
  "fmla   v21.4s ,  v14.4s,  v4.4s\n"    /* outr2 = w0 * r0, 4*/   \
  "fmla   v22.4s ,  v14.4s,  v6.4s\n"    /* outr3 = w0 * r0, 6*/   \
  "ldp    q8, q9,   [%[inr4]], #32\n"    /* load r0, 8-9 */        \
  "fmla   v19.4s ,  v15.4s,  v1.4s\n"    /* outr0 = w1 * r0, 1*/   \
  "fmla   v20.4s ,  v15.4s,  v3.4s\n"    /* outr1 = w1 * r0, 3*/   \
  "fmla   v21.4s ,  v15.4s,  v5.4s\n"    /* outr2 = w1 * r0, 5*/   \
  "fmla   v22.4s ,  v15.4s,  v7.4s\n"    /* outr3 = w1 * r0, 7*/   \
  "ldr    q10,   [%[inr4]] \n"           /* load r0, 10 */         \
  "fmla   v19.4s ,  v16.4s,  v2.4s\n"    /* outr0 = w0 * r0, 2*/   \
  "fmla   v20.4s ,  v16.4s,  v4.4s\n"    /* outr1 = w0 * r0, 4*/   \
  "fmla   v21.4s ,  v16.4s,  v6.4s\n"    /* outr2 = w0 * r0, 6*/   \
  "fmla   v22.4s ,  v16.4s,  v8.4s\n"    /* outr3 = w0 * r0, 8*/   \
  "sub    %[inr4], %[inr4], #32\n"       /* inr0 -= 32 */          \
  "fmla   v19.4s ,  v17.4s,  v3.4s\n"    /* outr0 = w3 * r1, 0*/   \
  "fmla   v20.4s ,  v17.4s,  v5.4s\n"    /* outr1 = w3 * r1, 2*/   \
  "fmla   v21.4s ,  v17.4s,  v7.4s\n"    /* outr2 = w3 * r1, 4*/   \
  "fmla   v22.4s ,  v17.4s,  v9.4s\n"    /* outr3 = w3 * r1, 6*/   \
  "fmla   v19.4s ,  v18.4s,  v4.4s\n"    /* outr0 = w3 * r1, 0*/   \
  "fmla   v20.4s ,  v18.4s,  v6.4s\n"    /* outr1 = w3 * r1, 2*/   \
  "fmla   v21.4s ,  v18.4s,  v8.4s\n"    /* outr2 = w3 * r1, 4*/   \
  "fmla   v22.4s ,  v18.4s,  v10.4s\n"   /* outr3 = w3 * r1, 6*/   \
  "sub    %[wc0], %[wc0], #320\n"        /* weight -= 320 */       \
  "trn1 v0.4s, v19.4s, v20.4s\n"         /* r0: a0a1c0c1*/         \
  "trn2 v1.4s, v19.4s, v20.4s\n"         /* r0: b0b1d0d1*/         \
  "trn1 v2.4s, v21.4s, v22.4s\n"         /* r0: a2a3c2c3*/         \
  "trn2 v3.4s, v21.4s, v22.4s\n"         /* r0: b2b3d2d3*/         \
  "trn1 v19.2d, v0.2d, v2.2d\n"          /* r0: a0a1a2a3*/         \
  "trn2 v21.2d, v0.2d, v2.2d\n"          /* r0: c0c1c2c3*/         \
  "trn1 v20.2d, v1.2d, v3.2d\n"          /* r0: b0b1b2b3*/         \
  "trn2 v22.2d, v1.2d, v3.2d\n"          /* r0: d0d1d2d3*/
#define RELU                             /* relu */     \
  "movi v0.4s, #0\n"                     /* for relu */ \
  "fmax v19.4s, v19.4s, v0.4s\n"                        \
  "fmax v20.4s, v20.4s, v0.4s\n"                        \
  "fmax v21.4s, v21.4s, v0.4s\n"                        \
  "fmax v22.4s, v22.4s, v0.4s\n"
#define RELU6 /* relu6 */             \
  "fmin v19.4s, v19.4s, %[vsix].4s\n" \
  "fmin v20.4s, v20.4s, %[vsix].4s\n" \
  "fmin v21.4s, v21.4s, %[vsix].4s\n" \
  "fmin v22.4s, v22.4s, %[vsix].4s\n"
#define LEAKY_RELU                       /* LeakyRelu */ \
  "movi v0.4s, #0\n"                     /* for relu */  \
  "fcmge v1.4s, v19.4s,  v0.4s \n"       /* vcgeq_f32 */ \
  "fmul  v2.4s, v19.4s, %[vscale].4s \n" /* mul */       \
  "fcmge v3.4s, v20.4s,  v0.4s \n"       /* vcgeq_f32 */ \
  "fmul  v4.4s, v20.4s, %[vscale].4s \n" /* mul */       \
  "fcmge v5.4s, v21.4s,  v0.4s \n"       /* vcgeq_f32 */ \
  "fmul  v6.4s, v21.4s, %[vscale].4s \n" /* mul */       \
  "fcmge v7.4s, v22.4s,  v0.4s \n"       /* vcgeq_f32 */ \
  "fmul  v8.4s, v22.4s, %[vscale].4s \n" /* mul */       \
  "bif  v19.16b, v2.16b, v1.16b \n"      /* choose*/     \
  "bif  v20.16b, v4.16b, v3.16b \n"      /* choose*/     \
  "bif  v21.16b, v6.16b, v5.16b \n"      /* choose*/     \
  "bif  v22.16b, v8.16b, v7.16b \n"      /* choose*/
#define STORE                            /* save result */ \
  "str q19, [%[outc0]], #16\n"                             \
  "str q20, [%[outc1]], #16\n"                             \
  "str q21, [%[outc2]], #16\n"                             \
  "str q22, [%[outc3]], #16\n"

#else
#define COMPUTE                                                              \
  /* fill with bias */                                                       \
  "vld1.32  {d12-d13}, [%[bias]]\n" /* load bias */ /* load weights */       \
  "vld1.32    {d14-d17}, [%[wc0]]!\n"               /* load w0-1, to q7-8 */ \
  "vld1.32  {d0-d3},   [%[r0]]!\n"                  /* load input r0, 0,1*/  \
  "vand.i32 q12,  q6, q6\n"                                                  \
  "vld1.32  {d4-d7},   [%[r0]]!\n" /* load input r0, 2,3*/                   \
  "vand.i32 q13,  q6, q6\n"                                                  \
  "vld1.32  {d8-d11},  [%[r0]]!\n" /* load input r0, 4,5*/                   \
  "vand.i32 q14,  q6, q6\n"                                                  \
  "vand.i32 q15,  q6, q6\n"                                                  \
  "vld1.32  {d12-d13}, [%[r0]]!\n" /* load input r0, 6*/                     \
  "vmla.f32   q12, q7, q0               @ w0 * inr0\n"                       \
  "vmla.f32   q13, q7, q2               @ w0 * inr2\n"                       \
  "vld1.32    {d18-d21}, [%[wc0]]!\n" /* load w2-3, to q9-q10 */             \
  "vmla.f32   q14, q7, q4               @ w0 * inr4\n"                       \
  "vmla.f32   q15, q7, q6               @ w0 * inr6\n"                       \
  "vmla.f32   q12, q8, q1              @ w1 * inr1\n"                        \
  "vmla.f32   q13, q8, q3              @ w1 * inr3\n"                        \
  "vmla.f32   q14, q8, q5              @ w1 * inr5\n"                        \
  "vld1.32    {d22-d23}, [%[wc0]]!\n" /* load w4, to q11 */                  \
  "vmla.f32   q12, q9, q2              @ w2 * inr2\n"                        \
  "vmla.f32   q13, q9, q4              @ w2 * inr6\n"                        \
  "vmla.f32   q14, q9, q6              @ w2 * inr4\n"                        \
  "vld1.32 {d0-d3}, [%[r0]]! \n" /* load r0, 7-8 */                          \
  "vmla.f32   q12, q10, q3              @ w3 * inr3\n"                       \
  "vmla.f32   q13, q10, q5              @ w3 * inr5\n"                       \
  "vmla.f32   q14, q10, q0              @ w3 * inr7\n"                       \
  "vmla.f32   q15, q8, q0               @ w1 * inr7\n"                       \
  "vld1.32 {d4-d7}, [%[r0]] \n" /* load r0, 9-10 */                          \
  "vmla.f32   q12, q11, q4              @ w4 * inr4\n"                       \
  "vmla.f32   q13, q11, q6              @ w4 * inr6\n"                       \
  "vmla.f32   q14, q11, q1              @ w4 * inr8\n"                       \
  "vmla.f32   q15, q9, q1               @ w2 * inr8\n"                       \
  "vld1.32    {d0-d3}, [%[r1]]!         @ load r1, 0, 1\n"                   \
  "vld1.32    {d14-d17}, [%[wc0]]!\n" /* load w0-1, to q7-8 */               \
  "vmla.f32   q15, q10, q2               @ w3 * inr9\n"                      \
  "vld1.32    {d4-d5}, [%[r1]]!         @ load r1, 2\n"                      \
  "sub %[r0], %[r0], #16             @ r0 - 16 to nextline address\n"        \
  "vld1.32    {d18-d21}, [%[wc0]]!\n" /* load w2-3, to q9-10 */              \
  "vmla.f32   q12, q7, q0              @ w0 * inr0\n"                        \
  "vmla.f32   q13, q7, q2              @ w0 * inr2\n"                        \
  "vmla.f32   q15, q11, q3               @ w4 * inr10\n"                     \
  "vld1.32    {d6-d9}, [%[r1]]!         @ load r1, 3, 4\n"                   \
  "vld1.32    {d22-d23}, [%[wc0]]!\n" /* load w4, to q11 */                  \
  "vld1.32    {d10-d13}, [%[r1]]!       @ load r1, 5, 6\n"                   \
  "vmla.f32   q14, q7, q4              @ w0 * inr0\n"                        \
  "vmla.f32   q15, q7, q6              @ w0 * inr2\n"                        \
  "vmla.f32   q12, q8, q1              @ w1 * inr1\n"                        \
  "vmla.f32   q13, q8, q3              @ w1 * inr3\n"                        \
  "vld1.32    {d0-d3}, [%[r1]]!         @ load r1, 7, 8\n"                   \
  "vmla.f32   q14, q8, q5              @ w1 * inr5\n"                        \
  "vmla.f32   q15, q8, q0              @ w1 * inr7\n"                        \
  "vmla.f32   q12, q9, q2              @ w2 * inr2\n"                        \
  "vmla.f32   q13, q9, q4              @ w2 * inr4\n"                        \
  "vmla.f32   q14, q9, q6              @ w2 * inr6\n"                        \
  "vmla.f32   q15, q9, q1              @ w2 * inr8\n"                        \
  "vmla.f32   q12, q10, q3              @ w3 * inr3\n"                       \
  "vld1.32    {d4-d7}, [%[r1]]         @ load r1, 9, 10\n"                   \
  "vmla.f32   q13, q10, q5              @ w3 * inr5\n"                       \
  "vmla.f32   q14, q10, q0              @ w3 * inr7\n"                       \
  "vmla.f32   q15, q10, q2              @ w3 * inr9\n"                       \
  "vld1.32    {d14-d17}, [%[wc0]]!\n" /* load w0-1, to q7-8 */               \
  "vmla.f32   q12, q11, q4              @ w4 * inr4\n"                       \
  "vmla.f32   q13, q11, q6              @ w4 * inr6\n"                       \
  "vmla.f32   q14, q11, q1              @ w4 * inr8\n"                       \
  "vmla.f32   q15, q11, q3              @ w4 * inr10\n"                      \
  "vld1.32    {d0-d3}, [%[r2]]!         @ load r2, 0, 1\n"                   \
  "vld1.32    {d18-d21}, [%[wc0]]!\n" /* load w2-3, to q9-10 */              \
  "sub %[r1], %[r1], #16                @ r1 - 16 to nextline address\n"     \
  "vld1.32    {d4-d7}, [%[r2]]!         @ load r2, 2, 3\n"                   \
  "vld1.32    {d22-d23}, [%[wc0]]!\n" /* load w4 to q11 */                   \
  "vmla.f32   q12, q7, q0              @ w0 * inr0\n"                        \
  "vmla.f32   q13, q7, q2              @ w0 * inr2\n"                        \
  "vld1.32    {d8-d11}, [%[r2]]!         @ load r2, 4, 5\n"                  \
  "vmla.f32   q12, q8, q1              @ w1 * inr1\n"                        \
  "vmla.f32   q13, q8, q3              @ w1 * inr3\n"                        \
  "vld1.32    {d12-d13}, [%[r2]]!         @ load r2, 6 \n"                   \
  "vmla.f32   q14, q7, q4              @ w0 * inr4\n"                        \
  "vmla.f32   q15, q7, q6              @ w0 * inr6\n"                        \
  "vld1.32    {d0-d3}, [%[r2]]!         @ load r2, 7, 8\n"                   \
  "vmla.f32   q12, q9, q2              @ w2 * inr2\n"                        \
  "vmla.f32   q13, q9, q4              @ w2 * inr4\n"                        \
  "vmla.f32   q14, q8, q5              @ w1 * inr5\n"                        \
  "vmla.f32   q15, q8, q0              @ w1 * inr7\n"                        \
  "vmla.f32   q12, q10, q3              @ w3 * inr3\n"                       \
  "vmla.f32   q13, q10, q5              @ w3 * inr5\n"                       \
  "vmla.f32   q14, q9, q6              @ w2 * inr6\n"                        \
  "vmla.f32   q15, q9, q1              @ w2 * inr8\n"                        \
  "vld1.32    {d4-d7}, [%[r2]]         @ load r2, 9, 10\n"                   \
  "vmla.f32   q12, q11, q4              @ w4 * inr4\n"                       \
  "vmla.f32   q13, q11, q6              @ w4 * inr6\n"                       \
  "vmla.f32   q14, q10, q0              @ w3 * inr7\n"                       \
  "vmla.f32   q15, q10, q2              @ w3 * inr9\n"                       \
  "vld1.32    {d14-d17}, [%[wc0]]!\n" /* load w0-1, to q7-8 */               \
  "sub %[r2], %[r2], #16                @ r1 - 16 to nextline address\n"     \
  "vmla.f32   q14, q11, q1              @ w4 * inr8\n"                       \
  "vld1.32    {d0-d3}, [%[r3]]!         @ load r3, 0, 1\n"                   \
  "vmla.f32   q15, q11, q3              @ w4 * inr10\n"                      \
  "vld1.32    {d4-d7}, [%[r3]]!         @ load r3, 2, 3\n"                   \
  "vld1.32    {d18-d21}, [%[wc0]]!\n" /* load w2-3, to q9-10 */              \
  "vmla.f32   q12, q7, q0              @ w0 * inr0\n"                        \
  "vmla.f32   q13, q7, q2              @ w0 * inr2\n"                        \
  "vld1.32    {d8-d11}, [%[r3]]!         @ load r3, 4, 5\n"                  \
  "vld1.32    {d22-d23}, [%[wc0]]!\n" /* load w4 to q11 */                   \
  "vld1.32    {d12-d13}, [%[r3]]!         @ load r3, 6, \n"                  \
  "vmla.f32   q12, q8, q1              @ w1 * inr1\n"                        \
  "vmla.f32   q13, q8, q3              @ w1 * inr3\n"                        \
  "vmla.f32   q14, q7, q4              @ w0 * inr4\n"                        \
  "vmla.f32   q15, q7, q6              @ w0 * inr6\n"                        \
  "vld1.32    {d0-d3}, [%[r3]]!         @ load r3, 7, 8\n"                   \
  "vmla.f32   q12, q9, q2              @ w2 * inr2\n"                        \
  "vmla.f32   q13, q9, q4              @ w2 * inr4\n"                        \
  "vmla.f32   q14, q8, q5              @ w1 * inr5\n"                        \
  "vmla.f32   q15, q8, q0              @ w1 * inr7\n"                        \
  "vmla.f32   q12, q10, q3              @ w3 * inr3\n"                       \
  "vld1.32    {d4-d7}, [%[r3]]         @ load r3, 9, 10\n"                   \
  "vmla.f32   q13, q10, q5              @ w3 * inr5\n"                       \
  "vmla.f32   q14, q9, q6              @ w2 * inr6\n"                        \
  "vmla.f32   q15, q9, q1              @ w2 * inr8\n"                        \
  "vmla.f32   q12, q11, q4              @ w4 * inr4\n"                       \
  "vmla.f32   q13, q11, q6              @ w4 * inr6\n"                       \
  "vmla.f32   q14, q10, q0              @ w3 * inr7\n"                       \
  "vmla.f32   q15, q10, q2              @ w3 * inr9\n"                       \
  "vld1.32    {d14-d17}, [%[wc0]]!\n" /* load w0-1, to q7-8 */               \
  "sub %[r3], %[r3], #16                @ r1 - 16 to nextline address\n"     \
  "vmla.f32   q14, q11, q1              @ w4 * inr8\n"                       \
  "vld1.32    {d0-d3}, [%[r4]]!         @ load r4, 0, 1\n"                   \
  "vmla.f32   q15, q11, q3              @ w4 * inr10\n"                      \
  "vld1.32    {d4-d7}, [%[r4]]!         @ load r4, 2, 3\n"                   \
  "vld1.32    {d18-d21}, [%[wc0]]!\n" /* load w2-3, to q9-10 */              \
  "vmla.f32   q12, q7, q0              @ w0 * inr0\n"                        \
  "vmla.f32   q13, q7, q2              @ w0 * inr2\n"                        \
  "vld1.32    {d8-d11}, [%[r4]]!         @ load r3, 4, 5\n"                  \
  "vld1.32    {d22-d23}, [%[wc0]]!\n" /* load w4 to q11 */                   \
  "vld1.32    {d12-d13}, [%[r4]]!         @ load r3, 6, \n"                  \
  "vmla.f32   q12, q8, q1              @ w1 * inr1\n"                        \
  "vmla.f32   q13, q8, q3              @ w1 * inr3\n"                        \
  "vmla.f32   q14, q7, q4              @ w0 * inr4\n"                        \
  "vmla.f32   q15, q7, q6              @ w0 * inr6\n"                        \
  "vld1.32    {d0-d3}, [%[r4]]!         @ load r3, 7, 8\n"                   \
  "vmla.f32   q12, q9, q2              @ w2 * inr2\n"                        \
  "vmla.f32   q13, q9, q4              @ w2 * inr4\n"                        \
  "vmla.f32   q14, q8, q5              @ w1 * inr5\n"                        \
  "vmla.f32   q15, q8, q0              @ w1 * inr7\n"                        \
  "vmla.f32   q12, q10, q3              @ w3 * inr3\n"                       \
  "vld1.32    {d4-d7}, [%[r4]]         @ load r3, 9, 10\n"                   \
  "vmla.f32   q13, q10, q5              @ w3 * inr5\n"                       \
  "vmla.f32   q14, q9, q6              @ w2 * inr6\n"                        \
  "vmla.f32   q15, q9, q1              @ w2 * inr8\n"                        \
  "vmla.f32   q12, q11, q4              @ w4 * inr4\n"                       \
  "vmla.f32   q13, q11, q6              @ w4 * inr6\n"                       \
  "vmla.f32   q14, q10, q0              @ w3 * inr7\n"                       \
  "vmla.f32   q15, q10, q2              @ w3 * inr9\n"                       \
  "sub    %[wc0], %[wc0], #400          @ wc0 - 400 to start address\n"      \
  "sub %[r4], %[r4], #16                @ r1 - 16 to nextline address\n"     \
  "vmla.f32   q14, q11, q1              @ w4 * inr8\n"                       \
  "vmla.f32   q15, q11, q3              @ w4 * inr10\n"                      \
  "vtrn.32 q12, q13\n" /* a0a1c0c1, b0b1d0d1*/                               \
  "vtrn.32 q14, q15\n" /* a2a3c2c3, b2b3d2d3*/                               \
  "vswp   d25, d28\n"  /* a0a1a2a3, c0c1c2c3*/                               \
  "vswp   d27, d30\n"  /* b0b1b2b3, d0d1d2d3*/

#define RELU /* relu */             \
  "vmov.u32 q0, #0\n"               \
  "vld1.32 {d2-d3}, [%[six_ptr]]\n" \
  "vmax.f32 q12, q12, q0\n"         \
  "vmax.f32 q13, q13, q0\n"         \
  "vmax.f32 q14, q14, q0\n"         \
  "vmax.f32 q15, q15, q0\n"
#define RELU6 /* relu6 */   \
  "vmin.f32 q12, q12, q1\n" \
  "vmin.f32 q13, q13, q1\n" \
  "vmin.f32 q14, q14, q1\n" \
  "vmin.f32 q15, q15, q1\n"
#define LEAKY_RELU /* LeakyRelu */    \
  "vmov.u32 q0, #0\n"                 \
  "vld1.32 {d2-d3}, [%[scale_ptr]]\n" \
  "vcge.f32 q2, q12, q0  @ q0 > 0 \n" \
  "vcge.f32 q4, q13, q0  @ q0 > 0 \n" \
  "vcge.f32 q6, q14, q0  @ q0 > 0 \n" \
  "vcge.f32 q8, q15, q0  @ q0 > 0 \n" \
  "vmul.f32 q3, q12, q1   @ mul \n"   \
  "vmul.f32 q5, q13, q1   @ mul \n"   \
  "vmul.f32 q7, q14, q1   @ mul \n"   \
  "vmul.f32 q9, q15, q1   @ mul \n"   \
  "vbif q12, q3, q2 @ choose \n"      \
  "vbif q13, q5, q4 @ choose \n"      \
  "vbif q14, q7, q6 @ choose \n"      \
  "vbif q15, q9, q8 @ choose \n"
#define STORE                        /* save result */ \
  "vst1.32 {d24-d25}, [%[outc0]]!\n" /* save outc0*/   \
  "vst1.32 {d26-d27}, [%[outc1]]!\n" /* save outc1*/   \
  "vst1.32 {d28-d29}, [%[outc2]]!\n" /* save outc2*/   \
  "vst1.32 {d30-d31}, [%[outc3]]!\n" /* save outc3*/

#endif

void act_switch_5x5s2(const float* inr0,
                      const float* inr1,
                      const float* inr2,
                      const float* inr3,
                      const float* inr4,
                      float* outc0,
                      float* outc1,
                      float* outc2,
                      float* outc3,
                      float32x4_t w0,
                      float32x4_t w1,
                      float32x4_t w2,
                      float32x4_t w3,
                      float32x4_t w4,
                      float32x4_t vbias,
                      const float* weight_c,
                      float* bias_local,
                      const operators::ActivationParam act_param) {
  bool has_active = act_param.has_active;
  if (has_active) {
    float tmp = act_param.Relu_clipped_coef;
    float ss = act_param.Leaky_relu_alpha;
#ifdef __aarch64__
    float32x4_t vsix = vdupq_n_f32(tmp);
    float32x4_t vscale = vdupq_n_f32(ss);
#else
    float vsix[4] = {tmp, tmp, tmp, tmp};
    float vscale[4] = {ss, ss, ss, ss};
#endif
    switch (act_param.active_type) {
      case lite_api::ActivationType::kRelu:
#ifdef __aarch64__
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
                     : [w0] "w"(w0),
                       [w1] "w"(w1),
                       [w2] "w"(w2),
                       [w3] "w"(w3),
                       [w4] "w"(w4),
                       [vbias] "w"(vbias)
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
                       "v14",
                       "v15",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20",
                       "v21",
                       "v22");
#else
        asm volatile(COMPUTE RELU STORE
                     : [r0] "+r"(inr0),
                       [r1] "+r"(inr1),
                       [r2] "+r"(inr2),
                       [r3] "+r"(inr3),
                       [r4] "+r"(inr4),
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
                     : [w0] "w"(w0),
                       [w1] "w"(w1),
                       [w2] "w"(w2),
                       [w3] "w"(w3),
                       [w4] "w"(w4),
                       [vbias] "w"(vbias),
                       [vsix] "w"(vsix)
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
                       "v14",
                       "v15",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20",
                       "v21",
                       "v22");
#else
        asm volatile(COMPUTE RELU RELU6 STORE
                     : [r0] "+r"(inr0),
                       [r1] "+r"(inr1),
                       [r2] "+r"(inr2),
                       [r3] "+r"(inr3),
                       [r4] "+r"(inr4),
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
                     : [w0] "w"(w0),
                       [w1] "w"(w1),
                       [w2] "w"(w2),
                       [w3] "w"(w3),
                       [w4] "w"(w4),
                       [vbias] "w"(vbias),
                       [vscale] "w"(vscale)
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
                       "v14",
                       "v15",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20",
                       "v21",
                       "v22");
#else
        asm volatile(COMPUTE LEAKY_RELU STORE
                     : [r0] "+r"(inr0),
                       [r1] "+r"(inr1),
                       [r2] "+r"(inr2),
                       [r3] "+r"(inr3),
                       [r4] "+r"(inr4),
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
                 : [w0] "w"(w0),
                   [w1] "w"(w1),
                   [w2] "w"(w2),
                   [w3] "w"(w3),
                   [w4] "w"(w4),
                   [vbias] "w"(vbias)
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
                   "v14",
                   "v15",
                   "v16",
                   "v17",
                   "v18",
                   "v19",
                   "v20",
                   "v21",
                   "v22");
#else
    asm volatile(COMPUTE STORE
                 : [r0] "+r"(inr0),
                   [r1] "+r"(inr1),
                   [r2] "+r"(inr2),
                   [r3] "+r"(inr3),
                   [r4] "+r"(inr4),
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

void conv_depthwise_5x5s2_fp32(const float* i_data,
                               float* o_data,
                               int bs,
                               int oc,
                               int oh,
                               int ow,
                               int ic,
                               int ih,
                               int win,
                               const float* weights,
                               const float* bias,
                               const operators::ConvParam& param,
                               const operators::ActivationParam act_param,
                               ARMContext* ctx) {
  auto paddings = *param.paddings;
  int threads = ctx->threads();
  const int pad_h = paddings[0];
  const int pad_w = paddings[2];
  const int out_c_block = 4;
  const int out_h_kernel = 1;
  const int out_w_kernel = 4;
  const int win_ext = ow * 2 + 3;
  const int ow_round = ROUNDUP(ow, 4);
  const int win_round = ROUNDUP(win_ext, 4);
  const int hin_round = oh * 2 + 3;
  const int prein_size = win_round * hin_round * out_c_block;
  auto workspace_size = threads * prein_size + win_round + ow_round;
  ctx->ExtendWorkspace(sizeof(float) * workspace_size);

  bool flag_bias = param.bias != nullptr;

  /// get workspace
  auto ptr_zero = ctx->workspace_data<float>();
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float* ptr_write = ptr_zero + win_round;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;

  int ws = -pad_w;
  int we = ws + win_round;
  int hs = -pad_h;
  int he = hs + hin_round;
  int w_loop = ow_round / 4;
  auto remain = w_loop * 4 - ow;
  bool flag_remain = remain > 0;
  remain = 4 - remain;
  remain = remain > 0 ? remain : 0;
  int row_len = win_round * out_c_block;

  float32x4_t vzero = vdupq_n_f32(0.f);

  for (int n = 0; n < bs; ++n) {
    const float* din_batch = i_data + n * ic * size_in_channel;
    float* dout_batch = o_data + n * oc * size_out_channel;
    LITE_PARALLEL_COMMON_BEGIN(c, tid, oc, 0, out_c_block) {
#ifdef LITE_USE_THREAD_POOL
      float* pre_din = ptr_write + ow_round + tid * prein_size;
#elif defined(ARM_WITH_OMP)
      float* pre_din = ptr_write + ow_round + omp_get_thread_num() * prein_size;
#else
      float* pre_din = ptr_write + ow_round;
#endif
      /// const array size
      prepack_input_nxwc4_dw(
          din_batch, pre_din, c, hs, he, ws, we, ic, win, ih, ptr_zero);
      const float* weight_c = weights + c * 25;  // kernel_w * kernel_h
      float* dout_c00 = dout_batch + c * size_out_channel;
      float bias_local[4] = {0, 0, 0, 0};

      if (flag_bias) {
        if (c + out_c_block < oc) {
          bias_local[0] = bias[c];
          bias_local[1] = bias[c + 1];
          bias_local[2] = bias[c + 2];
          bias_local[3] = bias[c + 3];
        } else {
          for (int k = 0; k < 4 && k + c < oc; k++) {
            bias_local[k] = bias[c + k];
          }
        }
      }
#ifdef __aarch64__
      float32x4_t w0 = vld1q_f32(weight_c);       // w0, v23
      float32x4_t w1 = vld1q_f32(weight_c + 4);   // w1, v24
      float32x4_t w2 = vld1q_f32(weight_c + 8);   // w2, v25
      float32x4_t w3 = vld1q_f32(weight_c + 12);  // w3, v26
      float32x4_t w4 = vld1q_f32(weight_c + 16);  // w4, v27
      float32x4_t vbias = vld1q_f32(bias_local);
      weight_c += 20;
#endif
      for (int h = 0; h < oh; h += out_h_kernel) {
        float* outc0 = dout_c00 + h * ow;
        float* outc1 = outc0 + size_out_channel;
        float* outc2 = outc1 + size_out_channel;
        float* outc3 = outc2 + size_out_channel;
        const float* inr0 = pre_din + h * 2 * row_len;
        const float* inr1 = inr0 + row_len;
        const float* inr2 = inr1 + row_len;
        const float* inr3 = inr2 + row_len;
        const float* inr4 = inr3 + row_len;

        if (c + out_c_block > oc) {
          switch (c + out_c_block - oc) {
            case 3:
              outc1 = ptr_write;
            case 2:
              outc2 = ptr_write;
            case 1:
              outc3 = ptr_write;
            default:
              break;
          }
        }
        auto c0 = outc0;
        auto c1 = outc1;
        auto c2 = outc2;
        auto c3 = outc3;
        float pre_out[16];
        for (int w = 0; w < w_loop; ++w) {
          bool flag_mask = (w == w_loop - 1) && flag_remain;
          if (flag_mask) {
            c0 = outc0;
            c1 = outc1;
            c2 = outc2;
            c3 = outc3;
            outc0 = pre_out;
            outc1 = pre_out + 4;
            outc2 = pre_out + 8;
            outc3 = pre_out + 12;
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
          act_switch_5x5s2(inr0,
                           inr1,
                           inr2,
                           inr3,
                           inr4,
                           outc0,
                           outc1,
                           outc2,
                           outc3,
                           vzero,
                           vzero,
                           vzero,
                           vzero,
                           vzero,
                           vzero,
                           weight_c,
                           bias_local,
                           act_param);
#endif
          if (flag_mask) {
            for (int i = 0; i < remain; ++i) {
              c0[i] = pre_out[i];
              c1[i] = pre_out[i + 4];
              c2[i] = pre_out[i + 8];
              c3[i] = pre_out[i + 12];
            }
          }
          inr0 += 32;
          inr1 += 32;
          inr2 += 32;
          inr3 += 32;
          inr4 += 32;
          outc0 += 4;
          outc1 += 4;
          outc2 += 4;
          outc3 += 4;
        }
      }
    }
    LITE_PARALLEL_END();
  }
}

#define ACTUAL_PARAM \
  dout, din, weights, bias, flag_bias, num, chin, hin, win, hout, wout

#define IN_PARAM                                                          \
  float *dout, const float *din, const float *weights, const float *bias, \
      bool flag_bias, int num, int chin, int hin, int win, int hout, int wout

#define DIN_PTR_INIT           \
  const float* din_ptr0 = dr0; \
  const float* din_ptr1 = dr1; \
  const float* din_ptr2 = dr2; \
  const float* din_ptr3 = dr3; \
  const float* din_ptr4 = dr4; \
  float* doutr0 = dout_ptr;    \
  /* h - 2 + 5 = h + 3 */      \
  if (h * 2 + 3 > hin) {       \
    switch (h * 2 + 3 - hin) { \
      case 4:                  \
        din_ptr1 = zero_ptr;   \
      case 3:                  \
        din_ptr2 = zero_ptr;   \
      case 2:                  \
        din_ptr3 = zero_ptr;   \
      case 1:                  \
        din_ptr4 = zero_ptr;   \
      default:                 \
        break;                 \
    }                          \
  }                            \
  /* update in_address */      \
  dr0 = dr2;                   \
  dr1 = dr3;                   \
  dr2 = dr4;                   \
  dr3 = dr2 + win;             \
  dr4 = dr3 + win;

// clang-format off
#ifdef __aarch64__
inline std::pair<uint32_t, uint32_t> right_mask_5x5s2p2_fp32(int win,
                                                             int wout,
                                                             uint32_t* vmask) {
  uint32_t right_pad_idx[20] = {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14,
                                9, 11, 13, 15, 16, 18, 17, 19};
  uint32_t cnt_col = ((wout >> 3) - 2);
  uint32_t size_right_remain = static_cast<uint32_t>(win - (14 + cnt_col * 16));
  if (size_right_remain >= 19) {
    cnt_col++;
    size_right_remain -= 16;
  }
  uint32_t cnt_remain = (size_right_remain >= 17 && wout % 8 == 0)
                            ? 8
                            : static_cast<uint32_t>(wout % 8);
  size_right_remain = (cnt_remain == 8) ? size_right_remain :
                      (size_right_remain + (8 - cnt_remain) * 2);
  uint32x4_t vmask_rp0 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx));
  uint32x4_t vmask_rp1 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx + 4));
  uint32x4_t vmask_rp2 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx + 8));
  uint32x4_t vmask_rp3 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx + 12));
  uint32x4_t vmask_rp4 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx + 16));
  vst1q_u32(vmask, vmask_rp0);
  vst1q_u32(vmask + 4,  vmask_rp1);
  vst1q_u32(vmask + 8,  vmask_rp2);
  vst1q_u32(vmask + 12, vmask_rp3);
  vst1q_u32(vmask + 16, vmask_rp4);
  return std::make_pair(cnt_col, cnt_remain);
}

#define LEFT_COMPUTE_S2                             \
  "PRFM PLDL1KEEP, [%[din_ptr0]]\n"                 \
  "PRFM PLDL1KEEP, [%[din_ptr1]]\n"                 \
  "PRFM PLDL1KEEP, [%[din_ptr2]]\n"                 \
  "PRFM PLDL1KEEP, [%[din_ptr3]]\n"                 \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr0]], #32\n"        \
  "ld2 {v4.4s, v5.4s}, [%[din_ptr1]], #32\n"        \
  "PRFM PLDL1KEEP, [%[din_ptr4]]\n"                 \
  "ld1 {v30.4s}, [%[bias_val]]\n"                   \
  "ld1 {v31.4s}, [%[bias_val]]\n"                   \
  "ld2 {v2.4s, v3.4s}, [%[din_ptr0]], #32\n"        \
  "ld2 {v6.4s, v7.4s}, [%[din_ptr1]], #32\n"        \
  "movi v28.4s, #0\n"                               \
  "movi v29.4s, #0\n"                               \
  "ld1  {v18.2s}, [%[din_ptr0]]\n"                  \
  "ld1  {v19.2s}, [%[din_ptr1]]\n"                  \
  /* line 0-1 */                                    \
  "fmla v30.4s, v0.4s, %[w0].s[2]\n"                \
  "fmla v31.4s, v2.4s, %[w0].s[2]\n"                \
  "fmla v28.4s, v4.4s, %[w1].s[3]\n"                \
  "fmla v29.4s, v6.4s, %[w1].s[3]\n"                \
  "ext  v8.16b,  %[vzero].16b, v0.16b, #12\n"       \
  "ext  v9.16b,  v0.16b,  v2.16b, #12\n"            \
  "ext  v14.16b, %[vzero].16b, v4.16b, #12\n"       \
  "ext  v15.16b, v4.16b,  v6.16b, #12\n"            \
  "fmla v30.4s, v1.4s, %[w0].s[3]\n"                \
  "fmla v31.4s, v3.4s, %[w0].s[3]\n"                \
  "fmla v28.4s, v5.4s, %[w2].s[0]\n"                \
  "fmla v29.4s, v7.4s, %[w2].s[0]\n"                \
  "ext  v10.16b, %[vzero].16b, v1.16b, #12\n"       \
  "ext  v11.16b, v1.16b,  v3.16b, #12\n"            \
  "ext  v16.16b, %[vzero].16b, v5.16b, #12\n"       \
  "ext  v17.16b, v5.16b,  v7.16b, #12\n"            \
  "fmla v30.4s,  v8.4s,  %[w0].s[0]\n"              \
  "fmla v31.4s,  v9.4s,  %[w0].s[0]\n"              \
  "fmla v28.4s,  v14.4s, %[w1].s[1]\n"              \
  "fmla v29.4s,  v15.4s, %[w1].s[1]\n"              \
  "ext  v12.16b, v0.16b, v2.16b, #4\n"              \
  "ext  v13.16b, v2.16b, v18.16b, #4\n"             \
  "ext  v8.16b,  v4.16b, v6.16b, #4\n"              \
  "ext  v9.16b,  v6.16b, v19.16b, #4\n"             \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr2]], #32\n"        \
  "ld2 {v4.4s, v5.4s}, [%[din_ptr3]], #32\n"        \
  "fmla v30.4s, v10.4s, %[w0].s[1]\n"               \
  "fmla v31.4s, v11.4s, %[w0].s[1]\n"               \
  "fmla v28.4s, v16.4s, %[w1].s[2]\n"               \
  "fmla v29.4s, v17.4s, %[w1].s[2]\n"               \
  "ld2 {v2.4s, v3.4s}, [%[din_ptr2]], #32\n"        \
  "ld2 {v6.4s, v7.4s}, [%[din_ptr3]], #32\n"        \
  "fmla v30.4s, v12.4s, %[w1].s[0]\n"               \
  "fmla v31.4s, v13.4s, %[w1].s[0]\n"               \
  "fmla v28.4s, v8.4s,  %[w2].s[1]\n"               \
  "fmla v29.4s, v9.4s,  %[w2].s[1]\n"               \
  "ld1  {v18.2s}, [%[din_ptr2]]\n"                  \
  "ld1  {v19.2s}, [%[din_ptr3]]\n"                  \
  /* line 2-3 */                                    \
  "fmla v30.4s, v0.4s, %[w3].s[0]\n"                \
  "fmla v31.4s, v2.4s, %[w3].s[0]\n"                \
  "fmla v28.4s, v4.4s, %[w4].s[1]\n"                \
  "fmla v29.4s, v6.4s, %[w4].s[1]\n"                \
  "ext  v8.16b,  %[vzero].16b, v0.16b, #12\n"       \
  "ext  v9.16b,  v0.16b,  v2.16b, #12\n"            \
  "ext  v14.16b, %[vzero].16b, v4.16b, #12\n"       \
  "ext  v15.16b, v4.16b,  v6.16b, #12\n"            \
  "fmla v30.4s, v1.4s, %[w3].s[1]\n"                \
  "fmla v31.4s, v3.4s, %[w3].s[1]\n"                \
  "fmla v28.4s, v5.4s, %[w4].s[2]\n"                \
  "fmla v29.4s, v7.4s, %[w4].s[2]\n"                \
  "ext  v10.16b, %[vzero].16b, v1.16b, #12\n"       \
  "ext  v11.16b, v1.16b,  v3.16b, #12\n"            \
  "ext  v16.16b, %[vzero].16b, v5.16b, #12\n"       \
  "ext  v17.16b, v5.16b,  v7.16b, #12\n"            \
  "fmla v30.4s,  v8.4s,  %[w2].s[2]\n"              \
  "fmla v31.4s,  v9.4s,  %[w2].s[2]\n"              \
  "fmla v28.4s,  v14.4s, %[w3].s[3]\n"              \
  "fmla v29.4s,  v15.4s, %[w3].s[3]\n"              \
  "ext  v12.16b, v0.16b, v2.16b, #4\n"              \
  "ext  v13.16b, v2.16b, v18.16b, #4\n"             \
  "ext  v8.16b,  v4.16b, v6.16b, #4\n"              \
  "ext  v9.16b,  v6.16b, v19.16b, #4\n"             \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr4]], #32\n"        \
  "fmla v30.4s, v10.4s, %[w2].s[3]\n"               \
  "fmla v31.4s, v11.4s, %[w2].s[3]\n"               \
  "fmla v28.4s, v16.4s, %[w4].s[0]\n"               \
  "fmla v29.4s, v17.4s, %[w4].s[0]\n"               \
  "ld2 {v2.4s, v3.4s}, [%[din_ptr4]], #32\n"        \
  "fmla v30.4s, v12.4s, %[w3].s[2]\n"               \
  "fmla v31.4s, v13.4s, %[w3].s[2]\n"               \
  "fmla v28.4s, v8.4s,  %[w4].s[3]\n"               \
  "fmla v29.4s, v9.4s,  %[w4].s[3]\n"               \
  "ld1  {v18.2s}, [%[din_ptr4]]\n"                  \
  "ext  v8.16b,  %[vzero].16b, v0.16b, #12\n"       \
  "ext  v9.16b,  v0.16b,  v2.16b, #12\n"            \
  "ext  v10.16b, %[vzero].16b, v1.16b, #12\n"       \
  "ext  v11.16b, v1.16b,  v3.16b, #12\n"            \
  "ext  v12.16b, v0.16b, v2.16b, #4\n"              \
  "ext  v13.16b, v2.16b, v18.16b, #4\n"             \
  /* line 4 */                                      \
  "sub  %[din_ptr0], %[din_ptr0], #8\n"             \
  "fmla v30.4s,  v8.4s,  %[w5].s[0]\n"              \
  "fmla v31.4s,  v9.4s,  %[w5].s[0]\n"              \
  "fmla v28.4s,  v10.4s, %[w5].s[1]\n"              \
  "fmla v29.4s,  v11.4s, %[w5].s[1]\n"              \
  "sub  %[din_ptr1], %[din_ptr1], #8\n"             \
  "fmla v30.4s,  v0.4s,  %[w5].s[2]\n"              \
  "fmla v31.4s,  v2.4s,  %[w5].s[2]\n"              \
  "fmla v28.4s,  v1.4s,  %[w5].s[3]\n"              \
  "fmla v29.4s,  v3.4s,  %[w5].s[3]\n"              \
  "sub  %[din_ptr2], %[din_ptr2], #8\n"             \
  "fmla v30.4s,  v12.4s, %[w6].s[0]\n"              \
  "fmla v31.4s,  v13.4s, %[w6].s[0]\n"              \
  "sub  %[din_ptr3], %[din_ptr3], #8\n"             \
  "sub  %[din_ptr4], %[din_ptr4], #8\n"

#define LEFT_RESULT_S2                              \
  "cmp %w[cnt], #16\n"                              \
  "fadd v28.4s, v28.4s, v30.4s\n"                   \
  "fadd v29.4s, v29.4s, v31.4s\n"                   \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr0]], #32\n"        \
  "ld2 {v4.4s, v5.4s}, [%[din_ptr1]], #32\n"        \
  "ld1 {v30.4s}, [%[bias_val]]\n"                   \
  "ld1 {v31.4s}, [%[bias_val]]\n"                   \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"        \
  "blt 2f                         \n"

#define LEFT_RESULT_S2_RELU                         \
  "fadd v28.4s, v28.4s, v30.4s\n"                   \
  "fadd v29.4s, v29.4s, v31.4s\n"                   \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr0]], #32\n"        \
  "ld2 {v4.4s, v5.4s}, [%[din_ptr1]], #32\n"        \
  "cmp %w[cnt], #16\n"                              \
  "fmax v28.4s, v28.4s, %[vzero].4s\n"              \
  "fmax v29.4s, v29.4s, %[vzero].4s\n"              \
  "ld1 {v30.4s}, [%[bias_val]]\n"                   \
  "ld1 {v31.4s}, [%[bias_val]]\n"                   \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"        \
  "blt 2f                         \n"

#define LEFT_RESULT_S2_RELU6                        \
  "ld1 {v2.4s}, [%[six_ptr]]   \n"                  \
  "fadd v28.4s, v28.4s, v30.4s\n"                   \
  "fadd v29.4s, v29.4s, v31.4s\n"                   \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr0]], #32\n"        \
  "ld2 {v4.4s, v5.4s}, [%[din_ptr1]], #32\n"        \
  "cmp %w[cnt], #16\n"                              \
  "fmax v28.4s, v28.4s, %[vzero].4s\n"              \
  "fmax v29.4s, v29.4s, %[vzero].4s\n"              \
  "ld1 {v30.4s}, [%[bias_val]]\n"                   \
  "ld1 {v31.4s}, [%[bias_val]]\n"                   \
  "fmin v28.4s, v28.4s, v2.4s\n"                    \
  "fmin v29.4s, v29.4s, v2.4s\n"                    \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"        \
  "blt 2f                         \n"

#define MID_COMPUTE_S2                              \
  "1:                        \n"                    \
  "ld2 {v2.4s, v3.4s}, [%[din_ptr0]], #32\n"        \
  "ld2 {v6.4s, v7.4s}, [%[din_ptr1]], #32\n"        \
  "movi v28.4s, #0\n"                               \
  "movi v29.4s, #0\n"                               \
  "fmla v30.4s, v0.4s,  %[w0].s[0]\n"               \
  "fmla v31.4s, v2.4s,  %[w0].s[0]\n"               \
  "ld2 {v12.2s, v13.2s}, [%[din_ptr0]]\n"           \
  "ld2 {v18.2s, v19.2s}, [%[din_ptr1]]\n"           \
  "fmla v28.4s, v4.4s,  %[w1].s[1]\n"               \
  "fmla v29.4s, v6.4s,  %[w1].s[1]\n"               \
  "ext v8.16b,  v0.16b,  v2.16b, #4\n"              \
  "ext v9.16b,  v2.16b,  v12.16b, #4\n"             \
  "ext v14.16b, v4.16b,  v6.16b, #4\n"              \
  "ext v15.16b, v6.16b,  v18.16b, #4\n"             \
  "fmla v30.4s, v1.4s,  %[w0].s[1]\n"               \
  "fmla v31.4s, v3.4s,  %[w0].s[1]\n"               \
  "fmla v28.4s, v5.4s,  %[w1].s[2]\n"               \
  "fmla v29.4s, v7.4s,  %[w1].s[2]\n"               \
  "ext v10.16b, v1.16b,  v3.16b, #4\n"              \
  "ext v11.16b, v3.16b,  v13.16b, #4\n"             \
  "ext v16.16b, v5.16b,  v7.16b, #4\n"              \
  "ext v17.16b, v7.16b,  v19.16b, #4\n"             \
  "fmla v30.4s, v8.4s,  %[w0].s[2]\n"               \
  "fmla v31.4s, v9.4s,  %[w0].s[2]\n"               \
  "fmla v28.4s, v14.4s, %[w1].s[3]\n"               \
  "fmla v29.4s, v15.4s, %[w1].s[3]\n"               \
  "ext v8.16b,  v0.16b,  v2.16b, #8\n"              \
  "ext v9.16b,  v2.16b,  v12.16b, #8\n"             \
  "ext v14.16b, v4.16b,  v6.16b, #8\n"              \
  "ext v15.16b, v6.16b,  v18.16b, #8\n"             \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr2]], #32\n"        \
  "ld2 {v4.4s, v5.4s}, [%[din_ptr3]], #32\n"        \
  "fmla v30.4s, v10.4s, %[w0].s[3]\n"               \
  "fmla v31.4s, v11.4s, %[w0].s[3]\n"               \
  "fmla v28.4s, v16.4s, %[w2].s[0]\n"               \
  "fmla v29.4s, v17.4s, %[w2].s[0]\n"               \
  "ld2 {v2.4s, v3.4s}, [%[din_ptr2]], #32\n"        \
  "ld2 {v6.4s, v7.4s}, [%[din_ptr3]], #32\n"        \
  "fmla v30.4s, v8.4s,  %[w1].s[0]\n"               \
  "fmla v31.4s, v9.4s,  %[w1].s[0]\n"               \
  "fmla v28.4s, v14.4s, %[w2].s[1]\n"               \
  "fmla v29.4s, v15.4s, %[w2].s[1]\n"               \
  "ld2 {v12.2s, v13.2s}, [%[din_ptr2]]\n"           \
  "ld2 {v18.2s, v19.2s}, [%[din_ptr3]]\n"           \
  /* line 2-3 */                                    \
  "fmla v30.4s, v0.4s,  %[w2].s[2]\n"               \
  "fmla v31.4s, v2.4s,  %[w2].s[2]\n"               \
  "fmla v28.4s, v4.4s,  %[w3].s[3]\n"               \
  "fmla v29.4s, v6.4s,  %[w3].s[3]\n"               \
  "ext v8.16b,  v0.16b,  v2.16b, #4\n"              \
  "ext v9.16b,  v2.16b,  v12.16b, #4\n"             \
  "ext v14.16b, v4.16b,  v6.16b, #4\n"              \
  "ext v15.16b, v6.16b,  v18.16b, #4\n"             \
  "fmla v30.4s, v1.4s,  %[w2].s[3]\n"               \
  "fmla v31.4s, v3.4s,  %[w2].s[3]\n"               \
  "fmla v28.4s, v5.4s,  %[w4].s[0]\n"               \
  "fmla v29.4s, v7.4s,  %[w4].s[0]\n"               \
  "ext v10.16b, v1.16b,  v3.16b, #4\n"              \
  "ext v11.16b, v3.16b,  v13.16b, #4\n"             \
  "ext v16.16b, v5.16b,  v7.16b, #4\n"              \
  "ext v17.16b, v7.16b,  v19.16b, #4\n"             \
  "fmla v30.4s, v8.4s,  %[w3].s[0]\n"               \
  "fmla v31.4s, v9.4s,  %[w3].s[0]\n"               \
  "fmla v28.4s, v14.4s, %[w4].s[1]\n"               \
  "fmla v29.4s, v15.4s, %[w4].s[1]\n"               \
  "ext v8.16b,  v0.16b,  v2.16b, #8\n"              \
  "ext v9.16b,  v2.16b,  v12.16b, #8\n"             \
  "ext v14.16b, v4.16b,  v6.16b, #8\n"              \
  "ext v15.16b, v6.16b,  v18.16b, #8\n"             \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr4]], #32\n"        \
  "fmla v30.4s, v10.4s, %[w3].s[1]\n"               \
  "fmla v31.4s, v11.4s, %[w3].s[1]\n"               \
  "fmla v28.4s, v16.4s, %[w4].s[2]\n"               \
  "fmla v29.4s, v17.4s, %[w4].s[2]\n"               \
  "ld2 {v2.4s, v3.4s}, [%[din_ptr4]], #32\n"        \
  "fmla v30.4s, v8.4s,  %[w3].s[2]\n"               \
  "fmla v31.4s, v9.4s,  %[w3].s[2]\n"               \
  "fmla v28.4s, v14.4s, %[w4].s[3]\n"               \
  "fmla v29.4s, v15.4s, %[w4].s[3]\n"               \
  "ld2 {v12.2s, v13.2s}, [%[din_ptr4]]\n"           \
  /* line 4 */                                      \
  "ext v8.16b,  v0.16b,  v2.16b, #4\n"              \
  "ext v9.16b,  v2.16b,  v12.16b, #4\n"             \
  "fmla v30.4s, v0.4s,  %[w5].s[0]\n"               \
  "fmla v31.4s, v2.4s,  %[w5].s[0]\n"               \
  "fmla v28.4s, v1.4s,  %[w5].s[1]\n"               \
  "fmla v29.4s, v3.4s,  %[w5].s[1]\n"               \
  "ext v10.16b, v1.16b,  v3.16b, #4\n"              \
  "ext v11.16b, v3.16b,  v13.16b, #4\n"             \
  "ext v14.16b, v0.16b,  v2.16b, #8\n"              \
  "ext v15.16b, v2.16b,  v12.16b, #8\n"             \
  "fmla v30.4s, v8.4s,  %[w5].s[2]\n"               \
  "fmla v31.4s, v9.4s,  %[w5].s[2]\n"               \
  "subs %w[cnt], %w[cnt], #16\n"                    \
  "fmla v28.4s, v10.4s, %[w5].s[3]\n"               \
  "fmla v29.4s, v11.4s, %[w5].s[3]\n"               \
  "fmla v30.4s, v14.4s, %[w6].s[0]\n"               \
  "fmla v31.4s, v15.4s, %[w6].s[0]\n"

#define MID_RESULT_S2                               \
  "cmp %w[cnt], #16\n"                              \
  "fadd v28.4s, v28.4s, v30.4s\n"                   \
  "fadd v29.4s, v29.4s, v31.4s\n"                   \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr0]], #32\n"        \
  "ld2 {v4.4s, v5.4s}, [%[din_ptr1]], #32\n"        \
  "ld1 {v30.4s}, [%[bias_val]]\n"                   \
  "ld1 {v31.4s}, [%[bias_val]]\n"                   \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"        \
  "bge 1b                         \n"

#define MID_RESULT_S2_RELU                          \
  "fadd v28.4s, v28.4s, v30.4s\n"                   \
  "fadd v29.4s, v29.4s, v31.4s\n"                   \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr0]], #32\n"        \
  "ld2 {v4.4s, v5.4s}, [%[din_ptr1]], #32\n"        \
  "cmp %w[cnt], #16\n"                              \
  "fmax v28.4s, v28.4s, %[vzero].4s\n"              \
  "fmax v29.4s, v29.4s, %[vzero].4s\n"              \
  "ld1 {v30.4s}, [%[bias_val]]\n"                   \
  "ld1 {v31.4s}, [%[bias_val]]\n"                   \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"        \
  "bge 1b                         \n"

#define MID_RESULT_S2_RELU6                         \
  "ld1 {v2.4s}, [%[six_ptr]]   \n"                  \
  "fadd v28.4s, v28.4s, v30.4s\n"                   \
  "fadd v29.4s, v29.4s, v31.4s\n"                   \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr0]], #32\n"        \
  "ld2 {v4.4s, v5.4s}, [%[din_ptr1]], #32\n"        \
  "cmp %w[cnt], #16\n"                              \
  "fmax v28.4s, v28.4s, %[vzero].4s\n"              \
  "fmax v29.4s, v29.4s, %[vzero].4s\n"              \
  "ld1 {v30.4s}, [%[bias_val]]\n"                   \
  "ld1 {v31.4s}, [%[bias_val]]\n"                   \
  "fmin v28.4s, v28.4s, v2.4s\n"                    \
  "fmin v29.4s, v29.4s, v2.4s\n"                    \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"        \
  "bge 1b                         \n"

#define RIGHT_COMPUTE_S2                            \
  "2:                             \n"               \
  "sub %[din_ptr0], %[din_ptr0], #32\n"             \
  "cmp %w[cnt], #1                \n"               \
  "sub %[din_ptr1], %[din_ptr1], #32\n"             \
  "sub %[din_ptr2], %[din_ptr2], %[right_pad_num_in]\n"\
  "sub %[din_ptr3], %[din_ptr3], %[right_pad_num_in]\n"\
  "sub %[din_ptr0], %[din_ptr0], %[right_pad_num_in]\n"\
  "sub %[din_ptr1], %[din_ptr1], %[right_pad_num_in]\n"\
  "blt 3f                         \n"               \
  "sub %[din_ptr4], %[din_ptr4], %[right_pad_num_in]\n"\
  "sub %[doutr0], %[doutr0], %[right_pad_num_out]\n"    \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr0]], #32\n"        \
  "ld2 {v6.4s, v7.4s}, [%[din_ptr1]], #32\n"        \
  "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[vmask]]\n"\
  "movi v28.4s, #0\n"                               \
  "movi v29.4s, #0\n"                               \
  "ld2 {v2.4s, v3.4s}, [%[din_ptr0]], #32\n"        \
  "ld2 {v8.4s, v9.4s}, [%[din_ptr1]], #32\n"        \
  "ldr q14, [%[vmask], #64]\n"                      \
  "ldr q15, [%[vmask], #72]\n"                      \
  "bif v0.16b, %[vzero].16b, v16.16b\n"             \
  "bif v6.16b, %[vzero].16b, v16.16b\n"             \
  "bif v1.16b, %[vzero].16b, v17.16b\n"             \
  "bif v7.16b, %[vzero].16b, v17.16b\n"             \
  "ld2 {v4.2s, v5.2s}, [%[din_ptr0]]\n"             \
  "ld2 {v10.2s, v11.2s}, [%[din_ptr1]]\n"           \
  "bif v2.16b, %[vzero].16b, v18.16b\n"             \
  "bif v8.16b, %[vzero].16b, v18.16b\n"             \
  "bif v3.16b, %[vzero].16b, v19.16b\n"             \
  "bif v9.16b, %[vzero].16b, v19.16b\n"             \
  "bif v4.16b, %[vzero].16b, v14.16b\n"             \
  "bif v10.16b,%[vzero].16b, v14.16b\n"             \
  "bif v5.16b, %[vzero].16b, v15.16b\n"             \
  "bif v11.16b,%[vzero].16b, v15.16b\n"             \
  /* line 0-1 */                                    \
  "fmla v30.4s, v0.4s,  %[w0].s[0]\n"               \
  "fmla v31.4s, v2.4s,  %[w0].s[0]\n"               \
  "fmla v28.4s, v6.4s,  %[w1].s[1]\n"               \
  "fmla v29.4s, v8.4s,  %[w1].s[1]\n"               \
  "ext v12.16b, v0.16b,  v2.16b, #4\n"              \
  "ext v13.16b, v2.16b,  v4.16b, #4\n"              \
  "ext v14.16b, v6.16b,  v8.16b, #4\n"              \
  "ext v15.16b, v8.16b,  v10.16b, #4\n"             \
  "fmla v30.4s, v1.4s,  %[w0].s[1]\n"               \
  "fmla v31.4s, v3.4s,  %[w0].s[1]\n"               \
  "fmla v28.4s, v7.4s,  %[w1].s[2]\n"               \
  "fmla v29.4s, v9.4s,  %[w1].s[2]\n"               \
  "fmla v30.4s, v12.4s, %[w0].s[2]\n"               \
  "fmla v31.4s, v13.4s, %[w0].s[2]\n"               \
  "fmla v28.4s, v14.4s, %[w1].s[3]\n"               \
  "fmla v29.4s, v15.4s, %[w1].s[3]\n"               \
  "ext v12.16b, v1.16b,  v3.16b, #4\n"              \
  "ext v13.16b, v3.16b,  v5.16b, #4\n"              \
  "ext v14.16b, v7.16b,  v9.16b, #4\n"              \
  "ext v15.16b, v9.16b,  v11.16b, #4\n"             \
  "fmla v30.4s, v12.4s, %[w0].s[3]\n"               \
  "fmla v31.4s, v13.4s, %[w0].s[3]\n"               \
  "fmla v28.4s, v14.4s, %[w2].s[0]\n"               \
  "fmla v29.4s, v15.4s, %[w2].s[0]\n"               \
  "ext v12.16b, v0.16b,  v2.16b, #8\n"              \
  "ext v13.16b, v2.16b,  v4.16b, #8\n"              \
  "ext v14.16b, v6.16b,  v8.16b, #8\n"              \
  "ext v15.16b, v8.16b,  v10.16b, #8\n"             \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr2]], #32\n"        \
  "ld2 {v6.4s, v7.4s}, [%[din_ptr3]], #32\n"        \
  "fmla v30.4s, v12.4s, %[w1].s[0]\n"               \
  "fmla v31.4s, v13.4s, %[w1].s[0]\n"               \
  "fmla v28.4s, v14.4s, %[w2].s[1]\n"               \
  "fmla v29.4s, v15.4s, %[w2].s[1]\n"               \
  "ld2 {v2.4s, v3.4s}, [%[din_ptr2]], #32\n"        \
  "ld2 {v8.4s, v9.4s}, [%[din_ptr3]], #32\n"        \
  "ldr q14, [%[vmask], #64]\n"                      \
  "ldr q15, [%[vmask], #72]\n"                      \
  "bif v0.16b, %[vzero].16b, v16.16b\n"             \
  "bif v6.16b, %[vzero].16b, v16.16b\n"             \
  "bif v1.16b, %[vzero].16b, v17.16b\n"             \
  "bif v7.16b, %[vzero].16b, v17.16b\n"             \
  "ld2 {v4.2s, v5.2s},   [%[din_ptr2]]\n"           \
  "ld2 {v10.2s, v11.2s}, [%[din_ptr3]]\n"           \
  "bif v2.16b, %[vzero].16b, v18.16b\n"             \
  "bif v8.16b, %[vzero].16b, v18.16b\n"             \
  "bif v3.16b, %[vzero].16b, v19.16b\n"             \
  "bif v9.16b, %[vzero].16b, v19.16b\n"             \
  "bif v4.16b, %[vzero].16b, v14.16b\n"             \
  "bif v10.16b,%[vzero].16b, v14.16b\n"             \
  "bif v5.16b, %[vzero].16b, v15.16b\n"             \
  "bif v11.16b,%[vzero].16b, v15.16b\n"             \
  /* line 2-3 */                                    \
  "fmla v30.4s, v0.4s,  %[w2].s[2]\n"               \
  "fmla v31.4s, v2.4s,  %[w2].s[2]\n"               \
  "fmla v28.4s, v6.4s,  %[w3].s[3]\n"               \
  "fmla v29.4s, v8.4s,  %[w3].s[3]\n"               \
  "ext v12.16b, v0.16b,  v2.16b, #4\n"              \
  "ext v13.16b, v2.16b,  v4.16b, #4\n"              \
  "ext v14.16b, v6.16b,  v8.16b, #4\n"              \
  "ext v15.16b, v8.16b,  v10.16b, #4\n"             \
  "fmla v30.4s, v1.4s,  %[w2].s[3]\n"               \
  "fmla v31.4s, v3.4s,  %[w2].s[3]\n"               \
  "fmla v28.4s, v7.4s,  %[w4].s[0]\n"               \
  "fmla v29.4s, v9.4s,  %[w4].s[0]\n"               \
  "fmla v30.4s, v12.4s, %[w3].s[0]\n"               \
  "fmla v31.4s, v13.4s, %[w3].s[0]\n"               \
  "fmla v28.4s, v14.4s, %[w4].s[1]\n"               \
  "fmla v29.4s, v15.4s, %[w4].s[1]\n"               \
  "ext v12.16b, v1.16b,  v3.16b, #4\n"              \
  "ext v13.16b, v3.16b,  v5.16b, #4\n"              \
  "ext v14.16b, v7.16b,  v9.16b, #4\n"              \
  "ext v15.16b, v9.16b,  v11.16b, #4\n"             \
  "fmla v30.4s, v12.4s, %[w3].s[1]\n"               \
  "fmla v31.4s, v13.4s, %[w3].s[1]\n"               \
  "fmla v28.4s, v14.4s, %[w4].s[2]\n"               \
  "fmla v29.4s, v15.4s, %[w4].s[2]\n"               \
  "ext v12.16b, v0.16b,  v2.16b, #8\n"              \
  "ext v13.16b, v2.16b,  v4.16b, #8\n"              \
  "ext v14.16b, v6.16b,  v8.16b, #8\n"              \
  "ext v15.16b, v8.16b,  v10.16b, #8\n"             \
  "ld2 {v0.4s, v1.4s}, [%[din_ptr4]], #32\n"        \
  "fmla v30.4s, v12.4s, %[w3].s[2]\n"               \
  "fmla v31.4s, v13.4s, %[w3].s[2]\n"               \
  "fmla v28.4s, v14.4s, %[w4].s[3]\n"               \
  "fmla v29.4s, v15.4s, %[w4].s[3]\n"               \
  "ld2 {v2.4s, v3.4s}, [%[din_ptr4]], #32\n"        \
  "ldr q14, [%[vmask], #64]\n"                      \
  "ldr q15, [%[vmask], #72]\n"                      \
  "bif v0.16b, %[vzero].16b, v16.16b\n"             \
  "bif v1.16b, %[vzero].16b, v17.16b\n"             \
  "ld2 {v4.2s, v5.2s},   [%[din_ptr4]]\n"           \
  "bif v2.16b, %[vzero].16b, v18.16b\n"             \
  "bif v3.16b, %[vzero].16b, v19.16b\n"             \
  "bif v4.16b, %[vzero].16b, v14.16b\n"             \
  "bif v5.16b, %[vzero].16b, v15.16b\n"             \
  /* line 4 */                                      \
  "ext v10.16b, v0.16b,  v2.16b, #4\n"              \
  "ext v11.16b, v2.16b,  v4.16b, #4\n"              \
  "ext v12.16b, v1.16b,  v3.16b, #4\n"              \
  "ext v13.16b, v3.16b,  v5.16b, #4\n"              \
  "fmla v30.4s, v0.4s,  %[w5].s[0]\n"               \
  "fmla v31.4s, v2.4s,  %[w5].s[0]\n"               \
  "fmla v28.4s, v1.4s,  %[w5].s[1]\n"               \
  "fmla v29.4s, v3.4s,  %[w5].s[1]\n"               \
  "ext v14.16b, v0.16b,  v2.16b, #8\n"              \
  "ext v15.16b, v2.16b,  v4.16b, #8\n"              \
  "fmla v30.4s, v10.4s, %[w5].s[2]\n"               \
  "fmla v31.4s, v11.4s, %[w5].s[2]\n"               \
  "fmla v28.4s, v12.4s, %[w5].s[3]\n"               \
  "fmla v29.4s, v13.4s, %[w5].s[3]\n"               \
  "fmla v30.4s, v14.4s, %[w6].s[0]\n"               \
  "fmla v31.4s, v15.4s, %[w6].s[0]\n"

#define RIGHT_RESULT_S2                             \
  "fadd v28.4s, v28.4s, v30.4s\n"                   \
  "fadd v29.4s, v29.4s, v31.4s\n"                   \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"        \
  "3:                             \n"

#define RIGHT_RESULT_S2_RELU                        \
  "fadd v28.4s, v28.4s, v30.4s\n"                   \
  "fadd v29.4s, v29.4s, v31.4s\n"                   \
  "fmax v28.4s, v28.4s, %[vzero].4s\n"              \
  "fmax v29.4s, v29.4s, %[vzero].4s\n"              \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"        \
  "3:                             \n"

#define RIGHT_RESULT_S2_RELU6                       \
  "ld1 {v2.4s}, [%[six_ptr]]   \n"                  \
  "fadd v28.4s, v28.4s, v30.4s\n"                   \
  "fadd v29.4s, v29.4s, v31.4s\n"                   \
  "fmax v28.4s, v28.4s, %[vzero].4s\n"              \
  "fmax v29.4s, v29.4s, %[vzero].4s\n"              \
  "fmin v28.4s, v28.4s, v2.4s\n"                    \
  "fmin v29.4s, v29.4s, v2.4s\n"                    \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"        \
  "3:                             \n"
#else
inline std::pair<uint32_t, uint32_t> right_mask_5x5s2p2_fp32(int win,
                                                             int wout,
                                                             uint32_t* vmask) {
  uint32_t right_pad_idx[12] = {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 9, 11};
  uint32_t cnt_col = ((wout >> 2) - 2);
  uint32_t size_right_remain = static_cast<uint32_t>(win - (6 + cnt_col * 8));
  if (size_right_remain >= 11) {
    cnt_col++;
    size_right_remain -= 8;
  }
  uint32_t cnt_remain = (size_right_remain >= 9 && wout % 4 == 0)
                            ? 4
                            : static_cast<uint32_t>(wout % 4);
  size_right_remain = (cnt_remain == 4) ? size_right_remain :
                      (size_right_remain + (4 - cnt_remain) * 2);
  uint32x4_t vmask_rp0 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx));
  uint32x4_t vmask_rp1 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx + 4));
  uint32x4_t vmask_rp2 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx + 8));
  vst1q_u32(vmask, vmask_rp0);
  vst1q_u32(vmask + 4, vmask_rp1);
  vst1q_u32(vmask + 8, vmask_rp2);
  return std::make_pair(cnt_col, cnt_remain);
}
#define LEFT_COMPUTE_S2                \
  "pld  [%[wei_ptr]]  \n"              \
  "pld  [%[din_ptr0]] \n"              \
  "pld  [%[din_ptr1]] \n"              \
  "pld  [%[din_ptr2]] \n"              \
  "pld  [%[din_ptr3]] \n"              \
  "vld1.32 {d0-d3}, [%[wei_ptr]]!\n"   \
  "pld  [%[din_ptr4]] \n"              \
  "vmov.u32  q7, #0   \n"              \
  "vld2.32 {d16-d19}, [%[din_ptr0]]!\n"\
  "vmov.u32  q14, #0   \n"             \
  "vld1.32 {d4-d7}, [%[wei_ptr]]!\n"   \
  "vld1.32 {d30-d31}, [%[bias_val]] \n"\
  /* line 0 */                         \
  "vld2.32 {d20-d21}, [%[din_ptr0]]\n" \
  "vld1.32 {d8-d11}, [%[wei_ptr]]!\n"  \
  "vext.32 q11, q7, q8,  #3\n"         \
  "vext.32 q12, q7, q9,  #3\n"         \
  "vmla.f32 q15, q8,  d1[0]\n"         \
  "vmla.f32 q14, q9,  d1[1]\n"         \
  "vext.32 q13, q8, q10, #1\n"         \
  "vld2.32 {d16-d19}, [%[din_ptr1]]!\n"\
  "vmla.f32 q15, q11, d0[0]\n"         \
  "vmla.f32 q14, q12, d0[1]\n"         \
  "vld1.32 {d12},     [%[wei_ptr]]!\n" \
  "sub      %[din_ptr0], #8\n"         \
  "vld2.32 {d20-d21}, [%[din_ptr1]]\n" \
  "vmla.f32 q15, q13, d2[0]\n"         \
  /* line 1 */                         \
  "vext.32 q11, q7, q8,  #3\n"         \
  "vext.32 q12, q7, q9,  #3\n"         \
  "vmla.f32 q14, q8,  d3[1]\n"         \
  "vmla.f32 q15, q9,  d4[0]\n"         \
  "vext.32 q13, q8, q10, #1\n"         \
  "vld2.32 {d16-d19}, [%[din_ptr2]]!\n"\
  "vmla.f32 q14, q11, d2[1]\n"         \
  "vmla.f32 q15, q12, d3[0]\n"         \
  "sub      %[din_ptr1], #8\n"         \
  "vld2.32 {d20-d21}, [%[din_ptr2]]\n" \
  "vmla.f32 q14, q13, d4[1]\n"         \
  /* line 2 */                         \
  "vext.32 q11, q7, q8,  #3\n"         \
  "vext.32 q12, q7, q9,  #3\n"         \
  "vmla.f32 q15, q8,  d6[0]\n"         \
  "vmla.f32 q14, q9,  d6[1]\n"         \
  "vext.32 q13, q8, q10, #1\n"         \
  "vld2.32 {d16-d19}, [%[din_ptr3]]!\n"\
  "vmla.f32 q15, q11, d5[0]\n"         \
  "vmla.f32 q14, q12, d5[1]\n"         \
  "sub      %[din_ptr2], #8\n"         \
  "vld2.32 {d20-d21}, [%[din_ptr3]]\n" \
  "vmla.f32 q15, q13, d7[0]\n"         \
  /* line 3 */                         \
  "vext.32 q11, q7, q8,  #3\n"         \
  "vext.32 q12, q7, q9,  #3\n"         \
  "vmla.f32 q14, q8,  d8[1]\n"         \
  "vmla.f32 q15, q9,  d9[0]\n"         \
  "vext.32 q13, q8, q10, #1\n"         \
  "vld2.32 {d16-d19}, [%[din_ptr4]]!\n"\
  "vmla.f32 q14, q11, d7[1]\n"         \
  "vmla.f32 q15, q12, d8[0]\n"         \
  "sub      %[din_ptr3], #8\n"         \
  "vld2.32 {d20-d21}, [%[din_ptr4]]\n" \
  "vmla.f32 q14, q13, d9[1]\n"         \
  /* line 4 */                         \
  "vext.32 q11, q7, q8,  #3\n"         \
  "vext.32 q12, q7, q9,  #3\n"         \
  "vmla.f32 q14, q8,  d11[0]\n"        \
  "vmla.f32 q15, q9,  d11[1]\n"        \
  "vext.32 q13, q8, q10, #1\n"         \
  "vld2.32 {d16-d19}, [%[din_ptr0]]!\n"\
  "vmla.f32 q14, q11, d10[0]\n"        \
  "vmla.f32 q15, q12, d10[1]\n"        \
  "sub      %[din_ptr4], #8\n"         \
  "vld2.32 {d20-d21}, [%[din_ptr0]]\n" \
  "vmla.f32 q14, q13, d12[0]\n"

#define LEFT_RESULT_S2                \
  "cmp %[cnt], #16\n"                 \
  "vadd.f32 q13, q14, q15\n"          \
  "vld1.32 {d30-d31}, [%[bias_val]]\n"\
  "vext.32  q11,  q8,  q10, #1\n"     \
  "vext.32  d24,  d18, d19, #1\n"     \
  "vext.32  d25,  d19, d21, #1\n"     \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "blt 2f\n"
#define LEFT_RESULT_S2_RELU           \
  "cmp %[cnt], #16\n"                 \
  "vadd.f32 q13, q14, q15\n"          \
  "vld1.32 {d30-d31}, [%[bias_val]]\n"\
  "vext.32  d22,  d16, d17, #1\n"     \
  "vext.32  d23,  d17, d20, #1\n"     \
  "vmax.f32 q13, q13, q7\n"           \
  "vext.32  d24,  d18, d19, #1\n"     \
  "vext.32  d25,  d19, d21, #1\n"     \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "blt 2f\n"
#define LEFT_RESULT_S2_RELU6          \
  "cmp %[cnt], #16\n"                 \
  "vadd.f32 q13, q14, q15\n"          \
  "vldr d28, [%[bias_val], #16]\n"    \
  "vldr d29, [%[bias_val], #24]\n"    \
  "vext.32  d22,  d16, d17, #1\n"     \
  "vext.32  d23,  d17, d20, #1\n"     \
  "vmax.f32 q13, q13, q7\n"           \
  "vld1.32 {d30-d31}, [%[bias_val]]\n"\
  "vext.32  d24,  d18, d19, #1\n"     \
  "vmin.f32 q13, q13, q14\n"          \
  "vext.32  d25,  d19, d21, #1\n"     \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "blt 2f\n"
#define MID_COMPUTE_S2                \
  "1:  \n"                            \
  "vmov.u32  q14, #0   \n"            \
  /* line 0 */                        \
  "vmla.f32 q15, q8,  d0[0]\n"        \
  "vmla.f32 q14, q9,  d0[1]\n"        \
  "vext.32  q13, q8,  q10, #2\n"      \
  "vld2.32 {d16-d19}, [%[din_ptr1]]!\n"\
  "vmla.f32 q15, q11, d1[0]\n"        \
  "vmla.f32 q14, q12, d1[1]\n"        \
  "vld2.32 {d20-d21}, [%[din_ptr1]]\n"\
  "vmla.f32 q15, q13, d2[0]\n"        \
  /* line 1 */                        \
  "vext.32  q11, q8,  q10, #1\n"      \
  "vext.32  d24, d18, d19, #1\n"      \
  "vext.32  d25, d19, d21, #1\n"      \
  "vmla.f32 q14, q8,  d2[1]\n"        \
  "vmla.f32 q15, q9,  d3[0]\n"        \
  "vext.32  q13, q8,  q10, #2\n"      \
  "vld2.32 {d16-d19}, [%[din_ptr2]]!\n"\
  "vmla.f32 q14, q11, d3[1]\n"        \
  "vmla.f32 q15, q12, d4[0]\n"        \
  "vld2.32 {d20-d21}, [%[din_ptr2]]\n"\
  "vmla.f32 q14, q13, d4[1]\n"        \
  /* line 2 */                        \
  "vext.32  q11, q8,  q10, #1\n"      \
  "vext.32  d24, d18, d19, #1\n"      \
  "vext.32  d25, d19, d21, #1\n"      \
  "vmla.f32 q15, q8,  d5[0]\n"        \
  "vmla.f32 q14, q9,  d5[1]\n"        \
  "vext.32  q13, q8,  q10, #2\n"      \
  "vld2.32 {d16-d19}, [%[din_ptr3]]!\n"\
  "vmla.f32 q15, q11, d6[0]\n"        \
  "vmla.f32 q14, q12, d6[1]\n"        \
  "vld2.32 {d20-d21}, [%[din_ptr3]]\n"\
  "vmla.f32 q15, q13, d7[0]\n"        \
  /* line 3 */                        \
  "vext.32  q11, q8,  q10, #1\n"      \
  "vext.32  d24, d18, d19, #1\n"      \
  "vext.32  d25, d19, d21, #1\n"      \
  "vmla.f32 q14, q8,  d7[1]\n"        \
  "vmla.f32 q15, q9,  d8[0]\n"        \
  "vext.32  q13, q8,  q10, #2\n"      \
  "vld2.32 {d16-d19}, [%[din_ptr4]]!\n"\
  "vmla.f32 q14, q11, d8[1]\n"        \
  "vmla.f32 q15, q12, d9[0]\n"        \
  "vld2.32 {d20-d21}, [%[din_ptr4]]\n"\
  "vmla.f32 q14, q13, d9[1]\n"        \
  /* line 5 */                        \
  "vext.32  q11, q8,  q10, #1\n"      \
  "vext.32  d24, d18, d19, #1\n"      \
  "vext.32  d25, d19, d21, #1\n"      \
  "vmla.f32 q15, q8,  d10[0]\n"       \
  "vmla.f32 q14, q9,  d10[1]\n"       \
  "vext.32  q13, q8,  q10, #2\n"      \
  "vld2.32 {d16-d19}, [%[din_ptr0]]!\n"\
  "vmla.f32 q15, q11, d11[0]\n"       \
  "vmla.f32 q14, q12, d11[1]\n"       \
  "sub      %[cnt],   #16\n"          \
  "vld2.32 {d20-d21}, [%[din_ptr0]]\n"\
  "vmla.f32 q15, q13, d12[0]\n"

#define MID_RESULT_S2                 \
  "vadd.f32 q13, q14, q15\n"          \
  "cmp     %[cnt], #16\n"             \
  "vld1.32 {d30-d31}, [%[bias_val]]\n"\
  "vext.32  q11,  q8,  q10, #1\n"     \
  "vext.32  d24,  d18, d19, #1\n"     \
  "vext.32  d25,  d19, d21, #1\n"     \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "bge 1b\n"
#define MID_RESULT_S2_RELU            \
  "vadd.f32 q13, q14, q15\n"          \
  "cmp     %[cnt], #16\n"             \
  "vld1.32 {d30-d31}, [%[bias_val]]\n"\
  "vext.32  q11,  q8,  q10, #1\n"     \
  "vmax.f32 q13,  q13, q7\n"          \
  "vext.32  d24,  d18, d19, #1\n"     \
  "vext.32  d25,  d19, d21, #1\n"     \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "bge 1b\n"
#define MID_RESULT_S2_RELU6           \
  "vadd.f32 q13, q14, q15\n"          \
  "vldr d28, [%[bias_val], #16]\n"    \
  "vldr d29, [%[bias_val], #24]\n"    \
  "cmp     %[cnt], #16\n"             \
  "vld1.32 {d30-d31}, [%[bias_val]]\n"\
  "vext.32  q11,  q8,  q10, #1\n"     \
  "vmax.f32 q13,  q13, q7\n"          \
  "vext.32  d24,  d18, d19, #1\n"     \
  "vext.32  d25,  d19, d21, #1\n"     \
  "vmin.f32 q13,  q13, q14\n"         \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "bge 1b\n"
#define RIGHT_COMPUTE_S2              \
  "2:  \n"                           \
  "sub     %[din_ptr0], #32\n"       \
  "cmp     %[cnt], #1\n"             \
  "vld1.32 {d22-d25}, [%[vmask]]\n"  \
  "sub     %[din_ptr0], %[right_pad_num_in]\n"\
  "sub     %[din_ptr1], %[right_pad_num_in]\n"\
  "sub     %[din_ptr2], %[right_pad_num_in]\n"\
  "blt 3f\n"                          \
  "vld2.32 {d16-d19}, [%[din_ptr0]]!\n"\
  "vldr    d26,       [%[vmask], #32]\n"\
  "vldr    d27,       [%[vmask], #40]\n"\
  "sub     %[doutr0], %[right_pad_num_out]\n"\
  "sub     %[din_ptr3], %[right_pad_num_in]\n"\
  "sub     %[din_ptr4], %[right_pad_num_in]\n"\
  "vld2.32 {d20-d21}, [%[din_ptr0]]\n"\
  "vbif q8, q7, q11\n"                \
  "vbif q9, q7, q12\n"                \
  "vld1.32 {d30-d31}, [%[bias_val]]\n"\
  "vmov.u32  q14, #0   \n"            \
  "vbif q10, q7, q13\n"               \
  /* line 0 */                        \
  "vext.32  q11, q8, q10, #1\n"       \
  "vmla.f32 q15, q8,  d0[0]\n"        \
  "vmla.f32 q14, q9,  d0[1]\n"        \
  "vext.32  d24, d18, d19, #1\n"      \
  "vext.32  d25, d19, d21, #1\n"      \
  "vmla.f32 q15, q11, d1[0]\n"        \
  "vmla.f32 q14, q12, d1[1]\n"        \
  "vext.32  q0,  q8,  q10, #2\n"      \
  "vld2.32 {d16-d19}, [%[din_ptr1]]!\n"\
  "vmla.f32 q15, q0,  d2[0]\n"        \
  /* line 1 */                        \
  "vld1.32 {d22-d25}, [%[vmask]]\n"   \
  "vld2.32 {d20-d21}, [%[din_ptr1]]\n"\
  "vbif q8, q7, q11\n"                \
  "vbif q9, q7, q12\n"                \
  "vbif q10, q7, q13\n"               \
  "vmla.f32 q14, q8,  d2[1]\n"        \
  "vmla.f32 q15, q9,  d3[0]\n"        \
  "vext.32  q11, q8,  q10, #1\n"      \
  "vext.32  d24, d18, d19, #1\n"      \
  "vext.32  d25, d19, d21, #1\n"      \
  "vext.32  q0,  q8,  q10, #2\n"      \
  "vld2.32 {d16-d19}, [%[din_ptr2]]!\n"\
  "vmla.f32 q14, q11, d3[1]\n"        \
  "vmla.f32 q15, q12, d4[0]\n"        \
  "vld1.32 {d22-d25}, [%[vmask]]\n"   \
  "vld2.32 {d20-d21}, [%[din_ptr2]]\n"\
  "vmla.f32 q14, q0, d4[1]\n"         \
  /* line 2 */                        \
  "vbif q8, q7, q11\n"                \
  "vbif q9, q7, q12\n"                \
  "vbif q10, q7, q13\n"               \
  "vext.32  q0,  q8,  q10, #1\n"      \
  "vext.32  d2,  d18, d19, #1\n"      \
  "vext.32  d3,  d19, d21, #1\n"      \
  "vmla.f32 q15, q8,  d5[0]\n"        \
  "vmla.f32 q14, q9,  d5[1]\n"        \
  "vext.32  q2,  q8,  q10, #2\n"      \
  "vld2.32 {d16-d19}, [%[din_ptr3]]!\n"\
  "vmla.f32 q15, q0,  d6[0]\n"        \
  "vmla.f32 q14, q1,  d6[1]\n"        \
  "vld2.32 {d20-d21}, [%[din_ptr3]]\n"\
  "vmla.f32 q15, q2,  d7[0]\n"        \
  /* line 3 */                        \
  "vbif q8, q7, q11\n"                \
  "vbif q9, q7, q12\n"                \
  "vbif q10, q7, q13\n"               \
  "vext.32  q0,  q8,  q10, #1\n"      \
  "vext.32  d2,  d18, d19, #1\n"      \
  "vext.32  d3,  d19, d21, #1\n"      \
  "vmla.f32 q14, q8,  d7[1]\n"        \
  "vmla.f32 q15, q9,  d8[0]\n"        \
  "vext.32  q2,  q8,  q10, #2\n"      \
  "vld2.32 {d16-d19}, [%[din_ptr4]]!\n"\
  "vmla.f32 q14, q0,  d8[1]\n"        \
  "vmla.f32 q15, q1,  d9[0]\n"        \
  "vld2.32 {d20-d21}, [%[din_ptr4]]\n"\
  "vmla.f32 q14, q2,  d9[1]\n"        \
  /* line 5 */                        \
  "vbif q8, q7, q11\n"                \
  "vbif q9, q7, q12\n"                \
  "vbif q10, q7, q13\n"               \
  "vext.32  q11, q8,  q10, #1\n"      \
  "vext.32  d24, d18, d19, #1\n"      \
  "vext.32  d25, d19, d21, #1\n"      \
  "vmla.f32 q15, q8,  d10[0]\n"       \
  "vmla.f32 q14, q9,  d10[1]\n"       \
  "vext.32  q13, q8,  q10, #2\n"      \
  "vmla.f32 q15, q11, d11[0]\n"       \
  "vmla.f32 q14, q12, d11[1]\n"       \
  "vmla.f32 q15, q13, d12[0]\n"
#define RIGHT_RESULT_S2               \
  "vadd.f32  q13, q15, q14\n"         \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "3:  \n"
#define RIGHT_RESULT_S2_RELU          \
  "vadd.f32  q13, q15, q14\n"         \
  "vmax.f32  q13, q13, q7\n"          \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "3:  \n"
#define RIGHT_RESULT_S2_RELU6         \
  "vadd.f32  q13, q15, q14\n"         \
  "vldr d28, [%[bias_val], #16]\n"    \
  "vldr d29, [%[bias_val], #24]\n"    \
  "vmax.f32  q13, q13, q7\n"          \
  "vmin.f32  q13, q13, q14\n"         \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "3:  \n"
#endif
// clang-format on
void conv_depthwise_5x5s2p2_fp32_relu(IN_PARAM, ARMContext* ctx) {
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 25;
  uint32_t vmask[20];
  auto&& res = right_mask_5x5s2p2_fp32(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
#ifdef __aarch64__
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 8);
  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
  float32x4_t vzero = vdupq_n_f32(0.f);
#else
  uint32_t right_pad_num_in = (cnt_remain == 4) ? 0 : ((4 - cnt_remain) * 8);
  uint32_t right_pad_num_out = (cnt_remain == 4) ? 0 : ((4 - cnt_remain) * 4);
#endif
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, (win + 16) * sizeof(float));
  float* write_ptr = zero_ptr + win + 16;
  cnt_col = (cnt_col << 4) + cnt_remain;
  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * chin * size_in_channel;
    float* dout_batch = dout + n * chin * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, chin) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const float* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0.f;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      const float* wei_ptr = weights + c * w_stride;
      const float* dr0 = zero_ptr;
      const float* dr1 = zero_ptr;
      const float* dr2 = din_ch_ptr;
      const float* dr3 = dr2 + win;
      const float* dr4 = dr3 + win;
#ifdef __aarch64__
      float32x4_t w0 = vld1q_f32(wei_ptr);
      float32x4_t w1 = vld1q_f32(wei_ptr + 4);
      float32x4_t w2 = vld1q_f32(wei_ptr + 8);
      float32x4_t w3 = vld1q_f32(wei_ptr + 12);
      float32x4_t w4 = vld1q_f32(wei_ptr + 16);
      float32x4_t w5 = vld1q_f32(wei_ptr + 20);
      float32x4_t w6 = vdupq_n_f32(wei_ptr[24]);
#endif
      for (int h = 0; h < hout; h++) {
        DIN_PTR_INIT
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(
            LEFT_COMPUTE_S2 LEFT_RESULT_S2_RELU MID_COMPUTE_S2
                MID_RESULT_S2_RELU RIGHT_COMPUTE_S2 RIGHT_RESULT_S2_RELU
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [doutr0] "+r"(doutr0),
              [cnt] "+r"(cnt)
            : [w0] "w"(w0),
              [w1] "w"(w1),
              [w2] "w"(w2),
              [w3] "w"(w3),
              [w4] "w"(w4),
              [w5] "w"(w5),
              [w6] "w"(w6),
              [vzero] "w"(vzero),
              [bias_val] "r"(vbias),
              [right_pad_num_in] "r"(right_pad_num_in),
              [right_pad_num_out] "r"(right_pad_num_out),
              [vmask] "r"(vmask)
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
              "v28",
              "v29",
              "v30",
              "v31");
#else
        auto weight_ptr = wei_ptr;
        asm volatile(
            LEFT_COMPUTE_S2 LEFT_RESULT_S2_RELU MID_COMPUTE_S2
                MID_RESULT_S2_RELU RIGHT_COMPUTE_S2 RIGHT_RESULT_S2_RELU
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [doutr0] "+r"(doutr0),
              [cnt] "+r"(cnt),
              [wei_ptr] "+r"(weight_ptr)
            : [bias_val] "r"(vbias),
              [right_pad_num_in] "r"(right_pad_num_in),
              [right_pad_num_out] "r"(right_pad_num_out),
              [vmask] "r"(vmask)
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
        dout_ptr += wout;
      }
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_5x5s2p2_fp32_relu6(IN_PARAM, float six, ARMContext* ctx) {
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 25;
  uint32_t vmask[20];
  auto&& res = right_mask_5x5s2p2_fp32(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
#ifdef __aarch64__
  uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 8);
  uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
  float32x4_t vzero = vdupq_n_f32(0.f);
  float six_ptr[4] = {six, six, six, six};
#else
  uint32_t right_pad_num_in = (cnt_remain == 4) ? 0 : ((4 - cnt_remain) * 8);
  uint32_t right_pad_num_out = (cnt_remain == 4) ? 0 : ((4 - cnt_remain) * 4);
#endif
  float* zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, (win + 16) * sizeof(float));
  float* write_ptr = zero_ptr + win + 16;
  cnt_col = (cnt_col << 4) + cnt_remain;
  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * chin * size_in_channel;
    float* dout_batch = dout + n * chin * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, chin) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const float* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0.f;
      const float* wei_ptr = weights + c * w_stride;
      const float* dr0 = zero_ptr;
      const float* dr1 = zero_ptr;
      const float* dr2 = din_ch_ptr;
      const float* dr3 = dr2 + win;
      const float* dr4 = dr3 + win;
#ifdef __aarch64__
      float32x4_t w0 = vld1q_f32(wei_ptr);
      float32x4_t w1 = vld1q_f32(wei_ptr + 4);
      float32x4_t w2 = vld1q_f32(wei_ptr + 8);
      float32x4_t w3 = vld1q_f32(wei_ptr + 12);
      float32x4_t w4 = vld1q_f32(wei_ptr + 16);
      float32x4_t w5 = vld1q_f32(wei_ptr + 20);
      float32x4_t w6 = vdupq_n_f32(wei_ptr[24]);
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
#else
      float vbias[8] = {
          bias_val, bias_val, bias_val, bias_val, six, six, six, six};
#endif
      for (int h = 0; h < hout; h++) {
        DIN_PTR_INIT
        int cnt = cnt_col;
#ifdef __aarch64__
        asm volatile(
            LEFT_COMPUTE_S2 LEFT_RESULT_S2_RELU6 MID_COMPUTE_S2
                MID_RESULT_S2_RELU6 RIGHT_COMPUTE_S2 RIGHT_RESULT_S2_RELU6
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [doutr0] "+r"(doutr0),
              [cnt] "+r"(cnt)
            : [w0] "w"(w0),
              [w1] "w"(w1),
              [w2] "w"(w2),
              [w3] "w"(w3),
              [w4] "w"(w4),
              [w5] "w"(w5),
              [w6] "w"(w6),
              [vzero] "w"(vzero),
              [bias_val] "r"(vbias),
              [six_ptr] "r"(six_ptr),
              [right_pad_num_in] "r"(right_pad_num_in),
              [right_pad_num_out] "r"(right_pad_num_out),
              [vmask] "r"(vmask)
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
              "v28",
              "v29",
              "v30",
              "v31");
#else
        auto weight_ptr = wei_ptr;
        asm volatile(
            LEFT_COMPUTE_S2 LEFT_RESULT_S2_RELU6 MID_COMPUTE_S2
                MID_RESULT_S2_RELU6 RIGHT_COMPUTE_S2 RIGHT_RESULT_S2_RELU6
            : [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [doutr0] "+r"(doutr0),
              [cnt] "+r"(cnt),
              [wei_ptr] "+r"(weight_ptr)
            : [bias_val] "r"(vbias),
              [right_pad_num_in] "r"(right_pad_num_in),
              [right_pad_num_out] "r"(right_pad_num_out),
              [vmask] "r"(vmask)
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
        dout_ptr += wout;
      }
    }
    LITE_PARALLEL_END();
  }
}
void conv_depthwise_5x5s2p2_fp32(float* dout,
                                 const float* din,
                                 const float* weights,
                                 const float* bias,
                                 bool flag_bias,
                                 int num,
                                 int chout,
                                 int hout,
                                 int wout,
                                 int chin,
                                 int hin,
                                 int win,
                                 const operators::ConvParam& param,
                                 ARMContext* ctx) {
  auto act_param = param.activation_param;
  bool has_active = act_param.has_active;
  auto act_type = act_param.active_type;
  if (has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      conv_depthwise_5x5s2p2_fp32_relu(ACTUAL_PARAM, ctx);
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      conv_depthwise_5x5s2p2_fp32_relu6(
          ACTUAL_PARAM, act_param.Relu_clipped_coef, ctx);
    } else {
      LOG(FATAL) << "this act_type: " << static_cast<int>(act_type)
                 << " fuse not support";
    }
  } else {
    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = 25;
    uint32_t vmask[20];
    auto&& res = right_mask_5x5s2p2_fp32(win, wout, vmask);
    uint32_t cnt_col = res.first;
    uint32_t cnt_remain = res.second;
#ifdef __aarch64__
    uint32_t right_pad_num_in = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 8);
    uint32_t right_pad_num_out = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
    float32x4_t vzero = vdupq_n_f32(0.f);
#else
    uint32_t right_pad_num_in = (cnt_remain == 4) ? 0 : ((4 - cnt_remain) * 8);
    uint32_t right_pad_num_out = (cnt_remain == 4) ? 0 : ((4 - cnt_remain) * 4);
#endif
    float* zero_ptr = ctx->workspace_data<float>();
    memset(zero_ptr, 0, (win + 16) * sizeof(float));
    float* write_ptr = zero_ptr + win + 16;
    cnt_col = (cnt_col << 4) + cnt_remain;
    for (int n = 0; n < num; ++n) {
      const float* din_batch = din + n * chin * size_in_channel;
      float* dout_batch = dout + n * chin * size_out_channel;
      LITE_PARALLEL_BEGIN(c, tid, chin) {
        float* dout_ptr = dout_batch + c * size_out_channel;
        const float* din_ch_ptr = din_batch + c * size_in_channel;
        float bias_val = flag_bias ? bias[c] : 0.f;
        float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
        const float* wei_ptr = weights + c * w_stride;
        const float* dr0 = zero_ptr;
        const float* dr1 = zero_ptr;
        const float* dr2 = din_ch_ptr;
        const float* dr3 = dr2 + win;
        const float* dr4 = dr3 + win;
#ifdef __aarch64__
        float32x4_t w0 = vld1q_f32(wei_ptr);
        float32x4_t w1 = vld1q_f32(wei_ptr + 4);
        float32x4_t w2 = vld1q_f32(wei_ptr + 8);
        float32x4_t w3 = vld1q_f32(wei_ptr + 12);
        float32x4_t w4 = vld1q_f32(wei_ptr + 16);
        float32x4_t w5 = vld1q_f32(wei_ptr + 20);
        float32x4_t w6 = vdupq_n_f32(wei_ptr[24]);
#endif
        for (int h = 0; h < hout; h++) {
          DIN_PTR_INIT
          int cnt = cnt_col;
#ifdef __aarch64__
          asm volatile(LEFT_COMPUTE_S2 LEFT_RESULT_S2 MID_COMPUTE_S2
                           MID_RESULT_S2 RIGHT_COMPUTE_S2 RIGHT_RESULT_S2
                       : [din_ptr0] "+r"(din_ptr0),
                         [din_ptr1] "+r"(din_ptr1),
                         [din_ptr2] "+r"(din_ptr2),
                         [din_ptr3] "+r"(din_ptr3),
                         [din_ptr4] "+r"(din_ptr4),
                         [doutr0] "+r"(doutr0),
                         [cnt] "+r"(cnt)
                       : [w0] "w"(w0),
                         [w1] "w"(w1),
                         [w2] "w"(w2),
                         [w3] "w"(w3),
                         [w4] "w"(w4),
                         [w5] "w"(w5),
                         [w6] "w"(w6),
                         [vzero] "w"(vzero),
                         [bias_val] "r"(vbias),
                         [right_pad_num_in] "r"(right_pad_num_in),
                         [right_pad_num_out] "r"(right_pad_num_out),
                         [vmask] "r"(vmask)
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
                         "v28",
                         "v29",
                         "v30",
                         "v31");
#else
          auto weight_ptr = wei_ptr;
          asm volatile(LEFT_COMPUTE_S2 LEFT_RESULT_S2 MID_COMPUTE_S2
                           MID_RESULT_S2 RIGHT_COMPUTE_S2 RIGHT_RESULT_S2
                       : [din_ptr0] "+r"(din_ptr0),
                         [din_ptr1] "+r"(din_ptr1),
                         [din_ptr2] "+r"(din_ptr2),
                         [din_ptr3] "+r"(din_ptr3),
                         [din_ptr4] "+r"(din_ptr4),
                         [doutr0] "+r"(doutr0),
                         [cnt] "+r"(cnt),
                         [wei_ptr] "+r"(weight_ptr)
                       : [bias_val] "r"(vbias),
                         [right_pad_num_in] "r"(right_pad_num_in),
                         [right_pad_num_out] "r"(right_pad_num_out),
                         [vmask] "r"(vmask)
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
          dout_ptr += wout;
        }
      }
      LITE_PARALLEL_END();
    }
  }
}
#undef LEFT_COMPUTE_S2
#undef LEFT_RESULT_S2
#undef LEFT_RESULT_S2_RELU
#undef LEFT_RESULT_S2_RELU6
#undef MID_COMPUTE_S2
#undef MID_RESULT_S2
#undef MID_RESULT_S2_RELU
#undef MID_RESULT_S2_RELU6
#undef RIGHT_RESULT_S2
#undef RIGHT_RESULT_S2_RELU
#undef RIGHT_RESULT_S2_RELU6
#undef DIN_PTR_INIT
#undef IN_PARAM
#undef ACTUAL_PARAM
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
