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
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/core/context.h"
#include "lite/operators/op_params.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {
#ifdef __aarch64__
#define COMPUTE                                                 \
  "ldr    q8, [%[bias]]\n"            /* load bias */           \
  "ldp    q0, q1,   [%[inr0]], #32\n" /* load input r0*/        \
  "and    v19.16b,  v8.16b, v8.16b\n"                           \
  "ldp    q2, q3,   [%[inr0]], #32\n" /* load input r0*/        \
  "and    v20.16b,  v8.16b, v8.16b\n"                           \
  "ldp    q4, q5,   [%[inr0]], #32\n" /* load input r0*/        \
  "and    v21.16b,  v8.16b, v8.16b\n"                           \
  "ldp    q6, q7,   [%[inr0]], #32\n" /* load input r0*/        \
  "and    v22.16b,  v8.16b, v8.16b\n"                           \
  "ldr    q8,       [%[inr0]]\n"        /* load input r0*/      \
  "fmla   v19.4s ,  %[w0].4s,  v0.4s\n" /* outr0 = w0 * r0, 0*/ \
  "fmla   v20.4s ,  %[w0].4s,  v2.4s\n" /* outr1 = w0 * r0, 2*/ \
  "fmla   v21.4s ,  %[w0].4s,  v4.4s\n" /* outr2 = w0 * r0, 4*/ \
  "fmla   v22.4s ,  %[w0].4s,  v6.4s\n" /* outr3 = w0 * r0, 6*/ \
  "fmla   v19.4s ,  %[w1].4s,  v1.4s\n" /* outr0 = w1 * r0, 1*/ \
  "ldp    q0, q1,   [%[inr1]], #32\n"   /* load input r1*/      \
  "fmla   v20.4s ,  %[w1].4s,  v3.4s\n" /* outr1 = w1 * r0, 3*/ \
  "fmla   v21.4s ,  %[w1].4s,  v5.4s\n" /* outr2 = w1 * r0, 5*/ \
  "fmla   v22.4s ,  %[w1].4s,  v7.4s\n" /* outr3 = w1 * r0, 7*/ \
  "fmla   v19.4s ,  %[w2].4s,  v2.4s\n" /* outr0 = w0 * r0, 2*/ \
  "ldp    q2, q3,   [%[inr1]], #32\n"   /* load input r1*/      \
  "fmla   v20.4s ,  %[w2].4s,  v4.4s\n" /* outr1 = w0 * r0, 4*/ \
  "ldp    q4, q5,   [%[inr1]], #32\n"   /* load input r1*/      \
  "fmla   v21.4s ,  %[w2].4s,  v6.4s\n" /* outr2 = w0 * r0, 6*/ \
  "ldp    q6, q7,   [%[inr1]], #32\n"   /* load input r1*/      \
  "fmla   v22.4s ,  %[w2].4s,  v8.4s\n" /* outr3 = w0 * r0, 8*/ \
  "ldr    q8,   [%[inr1]]\n"            /* load input r1*/      \
  "fmla   v19.4s ,  %[w3].4s,  v0.4s\n" /* outr0 = w3 * r1, 0*/ \
  "fmla   v20.4s ,  %[w3].4s,  v2.4s\n" /* outr1 = w3 * r1, 2*/ \
  "fmla   v21.4s ,  %[w3].4s,  v4.4s\n" /* outr2 = w3 * r1, 4*/ \
  "fmla   v22.4s ,  %[w3].4s,  v6.4s\n" /* outr3 = w3 * r1, 6*/ \
  "fmla   v19.4s ,  %[w4].4s,  v1.4s\n" /* outr0 = w4 * r1, 1*/ \
  "ldp    q0, q1,   [%[inr2]], #32\n"   /* load input r2*/      \
  "fmla   v20.4s ,  %[w4].4s,  v3.4s\n" /* outr1 = w4 * r1, 3*/ \
  "fmla   v21.4s ,  %[w4].4s,  v5.4s\n" /* outr2 = w4 * r1, 5*/ \
  "fmla   v22.4s ,  %[w4].4s,  v7.4s\n" /* outr3 = w4 * r1, 7*/ \
  "fmla   v19.4s ,  %[w5].4s,  v2.4s\n" /* outr0 = w5 * r1, 2*/ \
  "ldp    q2, q3,   [%[inr2]], #32\n"   /* load input r2*/      \
  "fmla   v20.4s ,  %[w5].4s,  v4.4s\n" /* outr1 = w5 * r1, 4*/ \
  "ldp    q4, q5,   [%[inr2]], #32\n"   /* load input r2*/      \
  "fmla   v21.4s ,  %[w5].4s,  v6.4s\n" /* outr2 = w5 * r1, 6*/ \
  "ldp    q6, q7,   [%[inr2]], #32\n"   /* load input r2*/      \
  "fmla   v22.4s ,  %[w5].4s,  v8.4s\n" /* outr3 = w5 * r1, 8*/ \
  "ldr    q8,   [%[inr2]]\n"            /* load input r2*/      \
  "fmla   v19.4s ,  %[w6].4s,  v0.4s\n" /* outr0 = w6 * r2, 0*/ \
  "fmla   v20.4s ,  %[w6].4s,  v2.4s\n" /* outr1 = w6 * r2, 2*/ \
  "fmla   v21.4s ,  %[w6].4s,  v4.4s\n" /* outr2 = w6 * r2, 4*/ \
  "fmla   v22.4s ,  %[w6].4s,  v6.4s\n" /* outr3 = w6 * r2, 6*/ \
  "fmla   v19.4s ,  %[w7].4s,  v1.4s\n" /* outr0 = w7 * r2, 1*/ \
  "fmla   v20.4s ,  %[w7].4s,  v3.4s\n" /* outr1 = w7 * r2, 3*/ \
  "fmla   v21.4s ,  %[w7].4s,  v5.4s\n" /* outr2 = w7 * r2, 5*/ \
  "fmla   v22.4s ,  %[w7].4s,  v7.4s\n" /* outr3 = w7 * r2, 7*/ \
  "fmla   v19.4s ,  %[w8].4s,  v2.4s\n" /* outr0 = w8 * r2, 2*/ \
  "fmla   v20.4s ,  %[w8].4s,  v4.4s\n" /* outr1 = w8 * r2, 4*/ \
  "fmla   v21.4s ,  %[w8].4s,  v6.4s\n" /* outr2 = w8 * r2, 6*/ \
  "fmla   v22.4s ,  %[w8].4s,  v8.4s\n" /* outr3 = w8 * r2, 8*/ \
  "trn1 v0.4s, v19.4s, v20.4s\n"        /* r0: a0a1c0c1*/       \
  "trn2 v1.4s, v19.4s, v20.4s\n"        /* r0: b0b1d0d1*/       \
  "trn1 v2.4s, v21.4s, v22.4s\n"        /* r0: a2a3c2c3*/       \
  "trn2 v3.4s, v21.4s, v22.4s\n"        /* r0: b2b3d2d3*/       \
  "trn1 v19.2d, v0.2d, v2.2d\n"         /* r0: a0a1a2a3*/       \
  "trn2 v21.2d, v0.2d, v2.2d\n"         /* r0: c0c1c2c3*/       \
  "trn1 v20.2d, v1.2d, v3.2d\n"         /* r0: b0b1b2b3*/       \
  "trn2 v22.2d, v1.2d, v3.2d\n"         /* r0: d0d1d2d3*/
#define RELU                            /* relu */     \
  "movi v0.4s, #0\n"                    /* for relu */ \
  "fmax v19.4s, v19.4s, v0.4s\n"                       \
  "fmax v20.4s, v20.4s, v0.4s\n"                       \
  "fmax v21.4s, v21.4s, v0.4s\n"                       \
  "fmax v22.4s, v22.4s, v0.4s\n"
#define RELU6 /* relu6 */             \
  "fmin v19.4s, v19.4s, %[vsix].4s\n" \
  "fmin v20.4s, v20.4s, %[vsix].4s\n" \
  "fmin v21.4s, v21.4s, %[vsix].4s\n" \
  "fmin v22.4s, v22.4s, %[vsix].4s\n"
#define LEAKY_RELU                      /* LeakyRelu */ \
  "movi v0.4s, #0\n"                    /* for relu */  \
  "fcmge v1.4s, v19.4s,  v0.4s \n"      /* vcgeq_u32 */ \
  "fmul v2.4s, v19.4s, %[vscale].4s \n" /* mul */       \
  "fcmge v3.4s, v20.4s,  v0.4s \n"      /* vcgeq_u32 */ \
  "fmul v4.4s, v20.4s, %[vscale].4s \n" /* mul */       \
  "fcmge v5.4s, v21.4s,  v0.4s \n"      /* vcgeq_u32 */ \
  "fmul v6.4s, v21.4s, %[vscale].4s \n" /* mul */       \
  "fcmge v7.4s, v22.4s,  v0.4s \n"      /* vcgeq_u32 */ \
  "fmul v8.4s, v22.4s, %[vscale].4s \n" /* mul */       \
  "bif  v19.16b, v2.16b, v1.16b \n"     /* choose*/     \
  "bif  v20.16b, v4.16b, v3.16b \n"     /* choose*/     \
  "bif  v21.16b, v6.16b, v5.16b \n"     /* choose*/     \
  "bif  v22.16b, v8.16b, v7.16b \n"     /* choose*/
#define STORE                           /* save result */ \
  "str q19, [%[outc0]], #16\n"                            \
  "str q20, [%[outc1]], #16\n"                            \
  "str q21, [%[outc2]], #16\n"                            \
  "str q22, [%[outc3]], #16\n"

#else
#define COMPUTE                                                                \
  /* fill with bias */                                                         \
  "vld1.32  {d16-d17}, [%[bias]]\n" /* load bias */ /* load weights */         \
  "vld1.32    {d18-d21}, [%[wc0]]!\n"               /* load w0-2, to q9-11 */  \
  "vld1.32  {d0-d3},   [%[r0]]!\n"                  /* load input r0, 0,1*/    \
  "vand.i32 q12,  q8, q8\n"                                                    \
  "vld1.32  {d4-d7},   [%[r0]]!\n" /* load input r0, 2,3*/                     \
  "vand.i32 q13,  q8, q8\n"                                                    \
  "vld1.32  {d8-d11},  [%[r0]]!\n" /* load input r0, 4,5*/                     \
  "vand.i32 q14,  q8, q8\n"                                                    \
  "vld1.32  {d12-d15}, [%[r0]]!\n" /* load input r0, 6,7*/                     \
  "vand.i32 q15,  q8, q8\n"                                                    \
  "vld1.32  {d16-d17}, [%[r0]]\n" /* load input r0, 8*/                        \
  "vmla.f32   q12, q9, q0               @ w0 * inr0\n"                         \
  "vmla.f32   q13, q9, q2               @ w0 * inr2\n"                         \
  "vld1.32    {d22-d23}, [%[wc0]]!\n" /* load w2, to q11 */                    \
  "vmla.f32   q14, q9, q4               @ w0 * inr4\n"                         \
  "vmla.f32   q15, q9, q6               @ w0 * inr6\n"                         \
  "vmla.f32   q12, q10, q1              @ w1 * inr1\n"                         \
  "vld1.32    {d0-d3}, [%[r1]]!         @ load r1, 0, 1\n"                     \
  "vmla.f32   q13, q10, q3              @ w1 * inr3\n"                         \
  "vmla.f32   q14, q10, q5              @ w1 * inr5\n"                         \
  "vmla.f32   q15, q10, q7              @ w1 * inr7\n"                         \
  "vld1.32    {d18-d21}, [%[wc0]]!\n" /* load w3-4, to q9-10 */                \
  "vmla.f32   q12, q11, q2              @ w2 * inr2\n"                         \
  "vld1.32    {d4-d7}, [%[r1]]!         @ load r1, 2, 3\n"                     \
  "vmla.f32   q13, q11, q4              @ w2 * inr4\n"                         \
  "vld1.32    {d8-d11}, [%[r1]]!        @ load r1, 4, 5\n"                     \
  "vmla.f32   q14, q11, q6              @ w2 * inr6\n"                         \
  "vld1.32    {d12-d15}, [%[r1]]!       @ load r1, 6, 7\n"                     \
  "vmla.f32   q15, q11, q8              @ w2 * inr8\n" /* mul r1 with w3, w4*/ \
  "vmla.f32   q12, q9, q0               @ w3 * inr0\n"                         \
  "vmla.f32   q13, q9, q2               @ w3 * inr2\n"                         \
  "vld1.32    {d22-d23}, [%[wc0]]!\n" /* load w5, to q11 */                    \
  "vmla.f32   q14, q9, q4               @ w3 * inr4\n"                         \
  "vmla.f32   q15, q9, q6               @ w3 * inr6\n"                         \
  "vld1.32    {d16-d17}, [%[r1]]\n" /* load input r1, 8*/                      \
  "vmla.f32   q12, q10, q1              @ w4 * inr1\n"                         \
  "vld1.32    {d0-d3}, [%[r2]]!         @ load r2, 0, 1\n"                     \
  "vmla.f32   q13, q10, q3              @ w4 * inr3\n"                         \
  "vmla.f32   q14, q10, q5              @ w4 * inr5\n"                         \
  "vmla.f32   q15, q10, q7              @ w4 * inr7\n"                         \
  "vld1.32    {d18-d21}, [%[wc0]]!\n" /* load w6-7, to q9-10 */                \
  "vmla.f32   q12, q11, q2              @ w5 * inr2\n"                         \
  "vld1.32    {d4-d7}, [%[r2]]!         @ load r2, 2, 3\n"                     \
  "vmla.f32   q13, q11, q4              @ w5 * inr4\n"                         \
  "vld1.32    {d8-d11}, [%[r2]]!        @ load r2, 4, 5\n"                     \
  "vmla.f32   q14, q11, q6              @ w5 * inr6\n"                         \
  "vld1.32    {d12-d15}, [%[r2]]!       @ load r2, 6, 7\n"                     \
  "vmla.f32   q15, q11, q8              @ w5 * inr8\n" /* mul r2 with w6, w7*/ \
  "vmla.f32   q12, q9, q0               @ w6 * inr0\n"                         \
  "vmla.f32   q13, q9, q2               @ w6 * inr2\n"                         \
  "vld1.32    {d22-d23}, [%[wc0]]!\n" /* load w8, to q11 */                    \
  "vmla.f32   q14, q9, q4               @ w6 * inr4\n"                         \
  "vmla.f32   q15, q9, q6               @ w6 * inr6\n"                         \
  "vld1.32    {d16-d17}, [%[r2]]\n" /* load input r2, 8*/                      \
  "vmla.f32   q12, q10, q1              @ w7 * inr1\n"                         \
  "vmla.f32   q13, q10, q3              @ w7 * inr3\n"                         \
  "vmla.f32   q14, q10, q5              @ w7 * inr5\n"                         \
  "vmla.f32   q15, q10, q7              @ w7 * inr7\n"                         \
  "sub    %[wc0], %[wc0], #144          @ wc0 - 144 to start address\n"        \
  "vmla.f32   q12, q11, q2              @ w8 * inr2\n"                         \
  "vmla.f32   q13, q11, q4              @ w8 * inr4\n"                         \
  "vmla.f32   q14, q11, q6              @ w8 * inr6\n"                         \
  "vmla.f32   q15, q11, q8              @ w8 * inr8\n" /* transpose */         \
  "vtrn.32 q12, q13\n"                                 /* a0a1c0c1, b0b1d0d1*/ \
  "vtrn.32 q14, q15\n"                                 /* a2a3c2c3, b2b3d2d3*/ \
  "vswp   d25, d28\n"                                  /* a0a1a2a3, c0c1c2c3*/ \
  "vswp   d27, d30\n"                                  /* b0b1b2b3, d0d1d2d3*/
#define RELU                                           /* relu */ \
  "vmov.u32 q0, #0\n"                                             \
  "vld1.32 {d2-d3}, [%[six_ptr]]\n"                               \
  "vmax.f32 q12, q12, q0\n"                                       \
  "vmax.f32 q13, q13, q0\n"                                       \
  "vmax.f32 q14, q14, q0\n"                                       \
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

void act_switch_3x3s2(const float* inr0,
                      const float* inr1,
                      const float* inr2,
                      float* outc0,
                      float* outc1,
                      float* outc2,
                      float* outc3,
                      const float* weight_c,
                      float* bias_local,
                      float32x4_t w0,
                      float32x4_t w1,
                      float32x4_t w2,
                      float32x4_t w3,
                      float32x4_t w4,
                      float32x4_t w5,
                      float32x4_t w6,
                      float32x4_t w7,
                      float32x4_t w8,
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
                       [outc0] "+r"(outc0),
                       [outc1] "+r"(outc1),
                       [outc2] "+r"(outc2),
                       [outc3] "+r"(outc3)
                     : [w0] "w"(w0),
                       [w1] "w"(w1),
                       [w2] "w"(w2),
                       [w3] "w"(w3),
                       [w4] "w"(w4),
                       [w5] "w"(w5),
                       [w6] "w"(w6),
                       [w7] "w"(w7),
                       [w8] "w"(w8),
                       [bias] "r"(bias_local)
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
                       "v19",
                       "v20",
                       "v21",
                       "v22");
#else
        asm volatile(COMPUTE RELU STORE
                     : [r0] "+r"(inr0),
                       [r1] "+r"(inr1),
                       [r2] "+r"(inr2),
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
                       [outc0] "+r"(outc0),
                       [outc1] "+r"(outc1),
                       [outc2] "+r"(outc2),
                       [outc3] "+r"(outc3)
                     : [w0] "w"(w0),
                       [w1] "w"(w1),
                       [w2] "w"(w2),
                       [w3] "w"(w3),
                       [w4] "w"(w4),
                       [w5] "w"(w5),
                       [w6] "w"(w6),
                       [w7] "w"(w7),
                       [w8] "w"(w8),
                       [bias] "r"(bias_local),
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
                       "v19",
                       "v20",
                       "v21",
                       "v22");
#else
        asm volatile(COMPUTE RELU RELU6 STORE
                     : [r0] "+r"(inr0),
                       [r1] "+r"(inr1),
                       [r2] "+r"(inr2),
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
                       [outc0] "+r"(outc0),
                       [outc1] "+r"(outc1),
                       [outc2] "+r"(outc2),
                       [outc3] "+r"(outc3)
                     : [w0] "w"(w0),
                       [w1] "w"(w1),
                       [w2] "w"(w2),
                       [w3] "w"(w3),
                       [w4] "w"(w4),
                       [w5] "w"(w5),
                       [w6] "w"(w6),
                       [w7] "w"(w7),
                       [w8] "w"(w8),
                       [bias] "r"(bias_local),
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
                       "v19",
                       "v20",
                       "v21",
                       "v22");
#else
        asm volatile(COMPUTE LEAKY_RELU STORE
                     : [r0] "+r"(inr0),
                       [r1] "+r"(inr1),
                       [r2] "+r"(inr2),
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
                   [outc0] "+r"(outc0),
                   [outc1] "+r"(outc1),
                   [outc2] "+r"(outc2),
                   [outc3] "+r"(outc3)
                 : [w0] "w"(w0),
                   [w1] "w"(w1),
                   [w2] "w"(w2),
                   [w3] "w"(w3),
                   [w4] "w"(w4),
                   [w5] "w"(w5),
                   [w6] "w"(w6),
                   [w7] "w"(w7),
                   [w8] "w"(w8),
                   [bias] "r"(bias_local)
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
                   "v19",
                   "v20",
                   "v21",
                   "v22");
#else
    asm volatile(COMPUTE STORE
                 : [r0] "+r"(inr0),
                   [r1] "+r"(inr1),
                   [r2] "+r"(inr2),
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

void conv_3x3s2_depthwise_fp32(const float* i_data,
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
  const int win_ext = ow * 2 + 1;
  const int ow_round = ROUNDUP(ow, 4);
  const int win_round = ROUNDUP(win_ext, 4);
  const int hin_round = oh * 2 + 1;
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
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < oc; c += out_c_block) {
#ifdef ARM_WITH_OMP
      float* pre_din = ptr_write + ow_round + omp_get_thread_num() * prein_size;
#else
      float* pre_din = ptr_write + ow_round;
#endif
      /// const array size
      prepack_input_nxwc4_dw(
          din_batch, pre_din, c, hs, he, ws, we, ic, win, ih, ptr_zero);
      const float* weight_c = weights + c * 9;  // kernel_w * kernel_h
      float* dout_c00 = dout_batch + c * size_out_channel;
      float bias_local[4] = {0, 0, 0, 0};
      if (flag_bias) {
        bias_local[0] = bias[c];
        bias_local[1] = bias[c + 1];
        bias_local[2] = bias[c + 2];
        bias_local[3] = bias[c + 3];
      }
#ifdef __aarch64__
      float32x4_t w0 = vld1q_f32(weight_c);       // w0, v23
      float32x4_t w1 = vld1q_f32(weight_c + 4);   // w1, v24
      float32x4_t w2 = vld1q_f32(weight_c + 8);   // w2, v25
      float32x4_t w3 = vld1q_f32(weight_c + 12);  // w3, v26
      float32x4_t w4 = vld1q_f32(weight_c + 16);  // w4, v27
      float32x4_t w5 = vld1q_f32(weight_c + 20);  // w5, v28
      float32x4_t w6 = vld1q_f32(weight_c + 24);  // w6, v29
      float32x4_t w7 = vld1q_f32(weight_c + 28);  // w7, v30
      float32x4_t w8 = vld1q_f32(weight_c + 32);  // w8, v31
#endif
      for (int h = 0; h < oh; h += out_h_kernel) {
        float* outc0 = dout_c00 + h * ow;
        float* outc1 = outc0 + size_out_channel;
        float* outc2 = outc1 + size_out_channel;
        float* outc3 = outc2 + size_out_channel;
        const float* inr0 = pre_din + h * 2 * row_len;
        const float* inr1 = inr0 + row_len;
        const float* inr2 = inr1 + row_len;
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
          act_switch_3x3s2(inr0,
                           inr1,
                           inr2,
                           outc0,
                           outc1,
                           outc2,
                           outc3,
                           weight_c,
                           bias_local,
                           w0,
                           w1,
                           w2,
                           w3,
                           w4,
                           w5,
                           w6,
                           w7,
                           w8,
                           act_param);
#else
          act_switch_3x3s2(inr0,
                           inr1,
                           inr2,
                           outc0,
                           outc1,
                           outc2,
                           outc3,
                           weight_c,
                           bias_local,
                           vzero,
                           vzero,
                           vzero,
                           vzero,
                           vzero,
                           vzero,
                           vzero,
                           vzero,
                           vzero,
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
          outc0 += 4;
          outc1 += 4;
          outc2 += 4;
          outc3 += 4;
        }
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
