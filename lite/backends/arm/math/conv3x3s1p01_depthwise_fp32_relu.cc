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
#include "lite/backends/arm/math/conv_depthwise.h"

#include "lite/core/parallel_defines.h"
namespace paddle {
namespace lite {
namespace arm {
namespace math {

#ifdef __aarch64__
#define INIT_S1                                                   \
  "PRFM PLDL1KEEP, [%[din_ptr0]] \n"                              \
  "PRFM PLDL1KEEP, [%[din_ptr1]] \n"                              \
  "PRFM PLDL1KEEP, [%[din_ptr2]] \n"                              \
  "PRFM PLDL1KEEP, [%[din_ptr3]] \n"                              \
  "PRFM PLDL1KEEP, [%[din_ptr4]] \n"                              \
  "PRFM PLDL1KEEP, [%[din_ptr5]] \n"                              \
  "movi   v21.4s, #0x0\n" /* out0 = 0 */                          \
                                                                  \
  "ld1 {v0.4s}, [%[din_ptr0]], #16   \n" /*vld1q_f32(din_ptr0)*/  \
  "ld1 {v2.4s}, [%[din_ptr1]], #16   \n" /*vld1q_f32(din_ptr0)*/  \
  "ld1 {v4.4s}, [%[din_ptr2]], #16   \n" /*vld1q_f32(din_ptr0)*/  \
  "ld1 {v6.4s}, [%[din_ptr3]], #16   \n" /*vld1q_f32(din_ptr0)*/  \
                                                                  \
  "ld1 {v1.4s}, [%[din_ptr0]]   \n" /*vld1q_f32(din_ptr0)*/       \
  "ld1 {v3.4s}, [%[din_ptr1]]   \n" /*vld1q_f32(din_ptr0)*/       \
  "ld1 {v5.4s}, [%[din_ptr2]]   \n" /*vld1q_f32(din_ptr0)*/       \
  "ld1 {v7.4s}, [%[din_ptr3]]   \n" /*vld1q_f32(din_ptr0)*/       \
                                                                  \
  "ld1 {v12.4s}, [%[bias_val]]     \n"  /*vdupq_n_f32(bias_val)*/ \
  "ld1 {v13.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/ \
  "ld1 {v14.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/ \
  "ld1 {v15.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/

#define LEFT_COMPUTE_S1                                                   \
  "ext  v16.16b, %[vzero].16b, v0.16b, #12 \n"           /* v16 = 00123*/ \
  "ext  v17.16b, v0.16b, v1.16b, #4 \n" /* v16 = 1234 */ /* r0 */         \
  "fmla v12.4s,  v0.4s,  %[w0].s[1]\n" /* outr00 += din0_0123 * w0[1]*/   \
                                                                          \
  "ld1 {v8.4s}, [%[din_ptr4]], #16   \n"  /*vld1q_f32(din_ptr0)*/         \
  "ld1 {v10.4s}, [%[din_ptr5]], #16   \n" /*vld1q_f32(din_ptr0)*/         \
  "sub %[din_ptr0], %[din_ptr0], #4 \n"   /* din_ptr0-- */                \
  "sub %[din_ptr1], %[din_ptr1], #4 \n"   /* din_ptr0-- */                \
                                                                          \
  "fmla v12.4s ,  v16.4s,  %[w0].s[0]\n" /* outr00 += din0_0012 * w0[0]*/ \
                                                                          \
  "ld1 {v9.4s}, [%[din_ptr4]]   \n"     /*vld1q_f32(din_ptr0)*/           \
  "ld1 {v11.4s}, [%[din_ptr5]]   \n"    /*vld1q_f32(din_ptr0)*/           \
  "sub %[din_ptr2], %[din_ptr2], #4 \n" /* din_ptr0-- */                  \
  "sub %[din_ptr3], %[din_ptr3], #4 \n" /* din_ptr0-- */                  \
                                                                          \
  "fmla v12.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_1234 * w0[2]*/ \
                                                                          \
  "ext  v16.16b, %[vzero].16b, v2.16b, #12 \n"           /* v16 = 00123*/ \
  "ext  v17.16b, v2.16b, v3.16b, #4 \n" /* v16 = 1234 */ /* r1 */         \
  "fmla v13.4s ,  v2.4s,  %[w0].s[1]\n" /* outr00 += din1_0123 * w0[1]*/  \
  "fmla v12.4s ,  v2.4s,  %[w1].s[1]\n" /* outr00 += din1_0123 * w1[1]*/  \
  "sub %[din_ptr4], %[din_ptr4], #4 \n" /* din_ptr0-- */                  \
  "sub %[din_ptr5], %[din_ptr5], #4 \n" /* din_ptr0-- */                  \
                                                                          \
  "fmla v13.4s ,  v16.4s,  %[w0].s[0]\n" /* outr00 += din1_0123 * w0[1]*/ \
  "fmla v12.4s ,  v16.4s,  %[w1].s[0]\n" /* outr00 += din1_0123 * w1[1]*/ \
                                                                          \
  "ld1 {v0.4s}, [%[din_ptr0]], #16   \n" /*vld1q_f32(din_ptr0)*/          \
  "ld1 {v2.4s}, [%[din_ptr1]], #16   \n" /*vld1q_f32(din_ptr0)*/          \
                                                                          \
  "fmla v13.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din1_0123 * w0[1]*/ \
  "fmla v12.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din1_0123 * w1[1]*/ \
                                                                          \
  "ext v17.16b, v4.16b, v5.16b, #4 \n"         /* v16=1234 */             \
  "ext  v16.16b, %[vzero].16b, v4.16b, #12 \n" /* v16 = 00123*/           \
                                                                          \
  /* r2 */                                                                \
  "fmla v14.4s ,  v4.4s,  %[w0].s[1]\n" /* outr00 += din2_0123 * w0[1]*/  \
  "fmla v13.4s ,  v4.4s,  %[w1].s[1]\n" /* outr00 += din2_0123 * w1[1]*/  \
  "fmla v12.4s ,  v4.4s,  %[w2].s[1]\n" /* outr00 += din2_0123 * w2[1]*/  \
                                                                          \
  "ld1 {v1.4s}, [%[din_ptr0]]   \n" /*vld1q_f32(din_ptr0)*/               \
  "ld1 {v3.4s}, [%[din_ptr1]]   \n" /*vld1q_f32(din_ptr0)*/               \
                                                                          \
  "fmla v14.4s ,  v16.4s,  %[w0].s[0]\n" /* outr00 += din2_0123 * w0[1]*/ \
  "fmla v13.4s ,  v16.4s,  %[w1].s[0]\n" /* outr00 += din2_0123 * w0[1]*/ \
  "fmla v12.4s ,  v16.4s,  %[w2].s[0]\n" /* outr00 += din2_0123 * w1[1]*/ \
                                                                          \
  "ld1 {v4.4s}, [%[din_ptr2]], #16   \n" /*vld1q_f32(din_ptr0)*/          \
                                                                          \
  "fmla v14.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din1_0123 * w0[1]*/ \
  "fmla v13.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din1_0123 * w0[1]*/ \
  "fmla v12.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din1_0123 * w1[1]*/ \
                                                                          \
  "ext  v16.16b, %[vzero].16b, v6.16b, #12 \n"           /* v16 = 00123*/ \
  "ext  v17.16b, v6.16b, v7.16b, #4 \n" /* v16 = 1234 */ /* r3 */         \
  "fmla v15.4s ,  v6.4s,  %[w0].s[1]\n" /*outr00 += din2_0123 * w0[1]*/   \
  "fmla v14.4s ,  v6.4s,  %[w1].s[1]\n" /* outr00 += din2_0123 * w1[1]*/  \
  "fmla v13.4s ,  v6.4s,  %[w2].s[1]\n" /* outr00 += din2_0123 * w2[1]*/  \
                                                                          \
  "ld1 {v6.4s}, [%[din_ptr3]], #16   \n" /*vld1q_f32(din_ptr0)*/          \
                                                                          \
  "fmla v15.4s ,  v16.4s,  %[w0].s[0]\n" /* outr00 += din2_0123 * w0[1]*/ \
  "fmla v14.4s ,  v16.4s,  %[w1].s[0]\n" /* outr00 += din2_0123 * w0[1]*/ \
  "fmla v13.4s ,  v16.4s,  %[w2].s[0]\n" /* outr00 += din2_0123 * w1[1]*/ \
                                                                          \
  "ld1 {v5.4s}, [%[din_ptr2]]   \n" /*vld1q_f32(din_ptr0)*/               \
  "ld1 {v7.4s}, [%[din_ptr3]]   \n" /*vld1q_f32(din_ptr0)*/               \
                                                                          \
  "fmla v15.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din1_0123 * w0[1]*/ \
  "fmla v14.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din1_0123 * w0[1]*/ \
  "fmla v13.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din1_0123 * w1[1]*/ \
                                                                          \
  "ext  v16.16b, %[vzero].16b, v8.16b, #12 \n" /* v16 = 00123*/           \
  "ext  v17.16b, v8.16b, v9.16b, #4 \n"        /* v16 = 1234 */

#define LEFT_RESULT_S1                                                      \
  /* r4 */                                                                  \
  "fmla v15.4s ,  v8.4s,  %[w1].s[1]\n" /* outr00 += din2_0123 * w1[1]*/    \
  "fmla v14.4s ,  v8.4s,  %[w2].s[1]\n" /* outr00 += din2_0123 * w2[1]*/    \
                                                                            \
  "st1 {v12.4s}, [%[doutr0]], #16 \n"    /* vst1q_f32() */                  \
  "st1 {v13.4s}, [%[doutr1]], #16 \n"    /* vst1q_f32() */                  \
  "ld1 {v8.4s}, [%[din_ptr4]], #16   \n" /*vld1q_f32(din_ptr0)*/            \
                                                                            \
  "fmla v15.4s ,  v16.4s,  %[w1].s[0]\n" /* outr00 += din2_0123 * w0[1]*/   \
  "fmla v14.4s ,  v16.4s,  %[w2].s[0]\n" /* outr00 += din2_0123 * w1[1]*/   \
                                                                            \
  "ld1 {v9.4s}, [%[din_ptr4]]   \n"     /*vld1q_f32(din_ptr0)*/             \
  "ld1 {v12.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/           \
  "ld1 {v13.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/           \
                                                                            \
  "fmla v15.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din1_0123 * w0[1]*/   \
  "fmla v14.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din1_0123 * w1[1]*/   \
                                                                            \
  "ext  v16.16b, %[vzero].16b, v10.16b, #12 \n"            /* v16 = 00123*/ \
  "ext  v17.16b, v10.16b, v11.16b, #4 \n" /* v16 = 1234 */ /* r5 */         \
  "fmla v15.4s ,  v10.4s,  %[w2].s[1]\n" /* outr00 += din2_0123 * w1[1]*/   \
                                                                            \
  "st1 {v14.4s}, [%[doutr2]], #16 \n"    /* vst1q_f32() */                  \
  "ld1 {v10.4s}, [%[din_ptr5]], #16  \n" /*vld1q_f32(din_ptr0)*/            \
                                                                            \
  "fmla v15.4s ,  v16.4s,  %[w2].s[0]\n" /* outr00 += din2_0123 * w0[1]*/   \
                                                                            \
  "ld1 {v11.4s}, [%[din_ptr5]]   \n"    /*vld1q_f32(din_ptr0)*/             \
  "ld1 {v14.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/           \
                                                                            \
  "fmla v15.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din1_0123 * w0[1]*/   \
                                                                            \
  "ext  v16.16b, v0.16b, v1.16b, #4 \n" /* v16 = 1234*/                     \
  "ext  v17.16b, v0.16b, v1.16b, #8 \n" /* v16 = 2345 */                    \
                                                                            \
  "st1 {v15.4s}, [%[doutr3]], #16 \n" /* vst1q_f32() */                     \
  "cmp  %w[cnt], #1                \n"                                      \
  "ld1 {v15.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/           \
                                                                            \
  "blt 3f                         \n"

#define MID_COMPUTE_S1                                                    \
  "1:                             \n"   /* r0 */                          \
  "fmla v12.4s ,  v0.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
                                                                          \
  "ld1 {v0.4s}, [%[din_ptr0]], #16   \n" /*vld1q_f32(din_ptr0)*/          \
                                                                          \
  "fmla v12.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/ \
                                                                          \
  "ld1 {v1.4s}, [%[din_ptr0]]   \n" /*vld1q_f32(din_ptr0)*/               \
                                                                          \
  "fmla v12.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
                                                                          \
  "ext  v16.16b, v2.16b, v3.16b, #4 \n"                  /* v16 = 1234*/  \
  "ext  v17.16b, v2.16b, v3.16b, #8 \n" /* v16 = 2345 */ /* r1 */         \
  "fmla v13.4s ,  v2.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
  "fmla v12.4s ,  v2.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
                                                                          \
  "ld1 {v2.4s}, [%[din_ptr1]], #16   \n" /*vld1q_f32(din_ptr0)*/          \
                                                                          \
  "fmla v13.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/ \
  "fmla v12.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/ \
                                                                          \
  "ld1 {v3.4s}, [%[din_ptr1]]   \n" /*vld1q_f32(din_ptr0)*/               \
                                                                          \
  "fmla v13.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
  "fmla v12.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
                                                                          \
  "ext  v16.16b, v4.16b, v5.16b, #4 \n"                  /* v16 = 1234*/  \
  "ext  v17.16b, v4.16b, v5.16b, #8 \n" /* v16 = 2345 */ /* r2 */         \
  "fmla v14.4s ,  v4.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
  "fmla v13.4s ,  v4.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
  "fmla v12.4s ,  v4.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
                                                                          \
  "ld1 {v4.4s}, [%[din_ptr2]], #16   \n" /*vld1q_f32(din_ptr0)*/          \
                                                                          \
  "fmla v14.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/ \
  "fmla v13.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/ \
  "fmla v12.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/ \
                                                                          \
  "ld1 {v5.4s}, [%[din_ptr2]]   \n" /*vld1q_f32(din_ptr0)*/               \
                                                                          \
  "fmla v14.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
  "fmla v13.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
  "fmla v12.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
                                                                          \
  "ext  v16.16b, v6.16b, v7.16b, #4 \n" /* v16 = 1234*/                   \
  "ext  v17.16b, v6.16b, v7.16b, #8 \n" /* v16 = 2345 */

#define MID_RESULT_S1                                                      \
  /* r3 */                                                                 \
  "fmla v15.4s ,  v6.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v14.4s ,  v6.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v13.4s ,  v6.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
                                                                           \
  "ld1 {v6.4s}, [%[din_ptr3]], #16   \n" /*vld1q_f32(din_ptr0)*/           \
  "st1 {v12.4s}, [%[doutr0]], #16     \n"                                  \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v14.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v13.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "ld1 {v7.4s}, [%[din_ptr3]]   \n"     /*vld1q_f32(din_ptr0)*/            \
  "ld1 {v12.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/          \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v14.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v13.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "ext  v16.16b, v8.16b, v9.16b, #4 \n"                  /* v16 = 1234*/   \
  "ext  v17.16b, v8.16b, v9.16b, #8 \n" /* v16 = 2345 */ /* r3 */          \
  "fmla v15.4s ,  v8.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v14.4s ,  v8.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
                                                                           \
  "ld1 {v8.4s}, [%[din_ptr4]], #16   \n" /*vld1q_f32(din_ptr0)*/           \
  "st1 {v13.4s}, [%[doutr1]], #16     \n"                                  \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v14.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "ld1 {v9.4s}, [%[din_ptr4]]   \n"     /*vld1q_f32(din_ptr0)*/            \
  "ld1 {v13.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/          \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v14.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "ext  v16.16b, v10.16b, v11.16b, #4 \n"                  /* v16 = 1234*/ \
  "ext  v17.16b, v10.16b, v11.16b, #8 \n" /* v16 = 2345 */ /* r3 */        \
  "fmla v15.4s ,  v10.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
                                                                           \
  "ld1 {v10.4s}, [%[din_ptr5]], #16   \n" /*vld1q_f32(din_ptr0)*/          \
  "st1 {v14.4s}, [%[doutr2]], #16     \n"                                  \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "ld1 {v11.4s}, [%[din_ptr5]]   \n"    /*vld1q_f32(din_ptr0)*/            \
  "ld1 {v14.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/          \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "ext  v16.16b, v0.16b, v1.16b, #4 \n" /* v16 = 1234*/                    \
  "ext  v17.16b, v0.16b, v1.16b, #8 \n" /* v16 = 2345 */                   \
                                                                           \
  "subs %w[cnt], %w[cnt], #1 \n"                                           \
                                                                           \
  "st1 {v15.4s}, [%[doutr3]], #16     \n"                                  \
  "ld1 {v15.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/          \
                                                                           \
  "bne 1b \n"

#define RIGHT_COMPUTE_S1                                                  \
  "3:                             \n"                                     \
                                                                          \
  "sub %[din_ptr0], %[din_ptr0], #16 \n"                                  \
  "sub %[din_ptr1], %[din_ptr1], #16 \n"                                  \
  "sub %[din_ptr2], %[din_ptr2], #16 \n"                                  \
  "sub %[din_ptr3], %[din_ptr3], #16 \n"                                  \
  "sub %[din_ptr4], %[din_ptr4], #16 \n"                                  \
  "sub %[din_ptr5], %[din_ptr5], #16 \n"                                  \
  "sub %[din_ptr0], %[din_ptr0], %[right_pad_num] \n"                     \
  "sub %[din_ptr1], %[din_ptr1], %[right_pad_num] \n"                     \
  "sub %[din_ptr2], %[din_ptr2], %[right_pad_num] \n"                     \
  "sub %[din_ptr3], %[din_ptr3], %[right_pad_num] \n"                     \
  "sub %[din_ptr4], %[din_ptr4], %[right_pad_num] \n"                     \
  "sub %[din_ptr5], %[din_ptr5], %[right_pad_num] \n"                     \
                                                                          \
  "ld1 {v0.4s, v1.4s}, [%[din_ptr0]] \n"                                  \
  "ld1 {v2.4s, v3.4s}, [%[din_ptr1]] \n"                                  \
  "ld1 {v4.4s, v5.4s}, [%[din_ptr2]] \n"                                  \
  "ld1 {v6.4s, v7.4s}, [%[din_ptr3]] \n"                                  \
  "ld1 {v8.4s, v9.4s}, [%[din_ptr4]] \n"                                  \
  "ld1 {v10.4s, v11.4s}, [%[din_ptr5]] \n"                                \
                                                                          \
  "ld1 {v19.4s}, [%[vmask]]         \n"                                   \
                                                                          \
  "sub %[doutr0], %[doutr0], %[right_pad_num] \n"                         \
  "sub %[doutr1], %[doutr1], %[right_pad_num] \n"                         \
  "sub %[doutr2], %[doutr2], %[right_pad_num] \n"                         \
  "sub %[doutr3], %[doutr3], %[right_pad_num] \n"                         \
                                                                          \
  "bif v1.16b, %[vzero].16b, v19.16b \n"                                  \
  "bif v3.16b, %[vzero].16b, v19.16b \n"                                  \
                                                                          \
  "bif v5.16b, %[vzero].16b, v19.16b \n"                                  \
  "bif v7.16b, %[vzero].16b, v19.16b \n"                                  \
                                                                          \
  "ext  v16.16b, v0.16b, v1.16b, #4 \n"                  /* v16 = 1234*/  \
  "ext  v17.16b, v0.16b, v1.16b, #8 \n" /* v16 = 2345 */ /* r0 */         \
  "fmla v12.4s,  v0.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
                                                                          \
  "bif v9.16b, %[vzero].16b, v19.16b \n"                                  \
  "bif v11.16b, %[vzero].16b, v19.16b \n"                                 \
                                                                          \
  "fmla v12.4s,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                          \
  "fmla v12.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
                                                                          \
  "ext  v16.16b, v2.16b, v3.16b, #4 \n"                  /* v16 = 1234*/  \
  "ext  v17.16b, v2.16b, v3.16b, #8 \n" /* v16 = 2345 */ /* r1 */         \
  "fmla v13.4s ,  v2.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
  "fmla v12.4s ,  v2.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
                                                                          \
  "fmla v13.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/ \
  "fmla v12.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/ \
                                                                          \
  "fmla v13.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
  "fmla v12.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
                                                                          \
  "ext  v16.16b, v4.16b, v5.16b, #4 \n"                  /* v16 = 1234*/  \
  "ext  v17.16b, v4.16b, v5.16b, #8 \n" /* v16 = 2345 */ /* r2 */         \
  "fmla v14.4s ,  v4.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
  "fmla v13.4s ,  v4.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
  "fmla v12.4s ,  v4.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
                                                                          \
  "fmla v14.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/ \
  "fmla v13.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/ \
  "fmla v12.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/ \
                                                                          \
  "fmla v14.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
  "fmla v13.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
  "fmla v12.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/ \
                                                                          \
  "ext  v16.16b, v6.16b, v7.16b, #4 \n" /* v16 = 1234*/                   \
  "ext  v17.16b, v6.16b, v7.16b, #8 \n" /* v16 = 2345 */

#define RIGHT_RESULT_S1                                                    \
  /* r3 */                                                                 \
  "fmla v15.4s ,  v6.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v14.4s ,  v6.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v13.4s ,  v6.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v14.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v13.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "st1 {v12.4s}, [%[doutr0]], #16     \n"                                  \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v14.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v13.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "ext  v16.16b, v8.16b, v9.16b, #4 \n"                  /* v16 = 1234*/   \
  "ext  v17.16b, v8.16b, v9.16b, #8 \n" /* v16 = 2345 */ /* r3 */          \
  "fmla v15.4s ,  v8.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v14.4s ,  v8.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v14.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "st1 {v13.4s}, [%[doutr1]], #16     \n"                                  \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v14.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "ext  v16.16b, v10.16b, v11.16b, #4 \n"                  /* v16 = 1234*/ \
  "ext  v17.16b, v10.16b, v11.16b, #8 \n" /* v16 = 2345 */ /* r3 */        \
  "fmla v15.4s ,  v10.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/  \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "st1 {v14.4s}, [%[doutr2]], #16     \n"                                  \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "st1 {v15.4s}, [%[doutr3]], #16     \n"

#define LEFT_RESULT_S1_RELU                                               \
  /* r4 */                                                                \
  "fmla v15.4s ,  v8.4s,  %[w1].s[1]\n" /* outr00 += din2_0123 * w1[1]*/  \
  "fmla v14.4s ,  v8.4s,  %[w2].s[1]\n" /* outr00 += din2_0123 * w2[1]*/  \
                                                                          \
  "fmax v12.4s, v12.4s, %[vzero].4s \n" /*relu*/                          \
  "fmax v13.4s, v13.4s, %[vzero].4s \n" /*relu*/                          \
                                                                          \
  "ld1 {v8.4s}, [%[din_ptr4]], #16   \n" /*vld1q_f32(din_ptr0)*/          \
                                                                          \
  "fmla v15.4s ,  v16.4s,  %[w1].s[0]\n" /* outr00 += din2_0123 * w0[1]*/ \
  "fmla v14.4s ,  v16.4s,  %[w2].s[0]\n" /* outr00 += din2_0123 * w1[1]*/ \
                                                                          \
  "st1 {v12.4s}, [%[doutr0]], #16 \n" /* vst1q_f32() */                   \
  "st1 {v13.4s}, [%[doutr1]], #16 \n" /* vst1q_f32() */                   \
                                                                          \
  "ld1 {v9.4s}, [%[din_ptr4]]   \n" /*vld1q_f32(din_ptr0)*/               \
                                                                          \
  "fmla v15.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din1_0123 * w0[1]*/ \
  "fmla v14.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din1_0123 * w1[1]*/ \
                                                                          \
  "ext  v16.16b, %[vzero].16b, v10.16b, #12 \n" /* v16 = 00123*/          \
  "ext  v17.16b, v10.16b, v11.16b, #4 \n"       /* v16 = 1234 */          \
  "ld1 {v12.4s}, [%[bias_val]]      \n"         /*vdupq_n_f32(bias_val)*/ \
  "ld1 {v13.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/ /* r5*/ \
  "fmla v15.4s ,  v10.4s,  %[w2].s[1]\n" /* outr00 += din2_0123 * w1[1]*/ \
                                                                          \
  "fmax v14.4s, v14.4s, %[vzero].4s \n" /*relu*/                          \
                                                                          \
  "ld1 {v10.4s}, [%[din_ptr5]], #16  \n" /*vld1q_f32(din_ptr0)*/          \
                                                                          \
  "fmla v15.4s ,  v16.4s,  %[w2].s[0]\n" /* outr00 += din2_0123 * w0[1]*/ \
                                                                          \
  "st1 {v14.4s}, [%[doutr2]], #16 \n" /* vst1q_f32() */                   \
                                                                          \
  "ld1 {v11.4s}, [%[din_ptr5]]   \n" /*vld1q_f32(din_ptr0)*/              \
                                                                          \
  "fmla v15.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din1_0123 * w0[1]*/ \
                                                                          \
  "ld1 {v14.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/         \
                                                                          \
  "ext  v16.16b, v0.16b, v1.16b, #4 \n" /* v16 = 1234*/                   \
  "ext  v17.16b, v0.16b, v1.16b, #8 \n" /* v16 = 2345 */                  \
                                                                          \
  "fmax v15.4s, v15.4s, %[vzero].4s \n" /*relu*/                          \
                                                                          \
  "st1 {v15.4s}, [%[doutr3]], #16 \n" /* vst1q_f32() */                   \
  "cmp  %w[cnt], #1                \n"                                    \
  "ld1 {v15.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/         \
  "blt 3f                         \n"

#define MID_RESULT_S1_RELU                                                 \
  /* r3 */                                                                 \
  "fmla v15.4s ,  v6.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v14.4s ,  v6.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v13.4s ,  v6.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
                                                                           \
  "ld1 {v6.4s}, [%[din_ptr3]], #16   \n" /*vld1q_f32(din_ptr0)*/           \
  "fmax v12.4s, v12.4s, %[vzero].4s \n"  /*relu*/                          \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v14.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v13.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "st1 {v12.4s}, [%[doutr0]], #16     \n"                                  \
                                                                           \
  "ld1 {v7.4s}, [%[din_ptr3]]   \n"     /*vld1q_f32(din_ptr0)*/            \
  "ld1 {v12.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/          \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v14.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v13.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "ext  v16.16b, v8.16b, v9.16b, #4 \n"                  /* v16 = 1234*/   \
  "ext  v17.16b, v8.16b, v9.16b, #8 \n" /* v16 = 2345 */ /* r3 */          \
  "fmla v15.4s ,  v8.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v14.4s ,  v8.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
                                                                           \
  "ld1 {v8.4s}, [%[din_ptr4]], #16   \n" /*vld1q_f32(din_ptr0)*/           \
  "fmax v13.4s, v13.4s, %[vzero].4s \n"  /*relu*/                          \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v14.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "st1 {v13.4s}, [%[doutr1]], #16     \n"                                  \
                                                                           \
  "ld1 {v9.4s}, [%[din_ptr4]]   \n"     /*vld1q_f32(din_ptr0)*/            \
  "ld1 {v13.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/          \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v14.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "ext  v16.16b, v10.16b, v11.16b, #4 \n" /* v16 = 1234*/                  \
  "ext  v17.16b, v10.16b, v11.16b, #8 \n" /* v16 = 2345 */                 \
                                                                           \
  /* r3 */                                                                 \
  "fmla v15.4s ,  v10.4s,  %[w2].s[0]\n"  /* outr00 += din0_0123 * w0[0]*/ \
  "ld1 {v10.4s}, [%[din_ptr5]], #16   \n" /*vld1q_f32(din_ptr0)*/          \
  "fmax v14.4s, v14.4s, %[vzero].4s \n"   /*relu*/                         \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "st1 {v14.4s}, [%[doutr2]], #16     \n"                                  \
                                                                           \
  "ld1 {v11.4s}, [%[din_ptr5]]   \n"    /*vld1q_f32(din_ptr0)*/            \
  "ld1 {v14.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/          \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "ext  v16.16b, v0.16b, v1.16b, #4 \n" /* v16 = 1234*/                    \
  "ext  v17.16b, v0.16b, v1.16b, #8 \n" /* v16 = 2345 */                   \
                                                                           \
  "subs %w[cnt], %w[cnt], #1 \n"                                           \
                                                                           \
  "fmax v15.4s, v15.4s, %[vzero].4s \n" /*relu*/                           \
                                                                           \
  "st1 {v15.4s}, [%[doutr3]], #16     \n"                                  \
  "ld1 {v15.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/          \
                                                                           \
  "bne 1b \n"

#define RIGHT_RESULT_S1_RELU                                               \
  /* r3 */                                                                 \
  "fmla v15.4s ,  v6.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v14.4s ,  v6.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v13.4s ,  v6.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
                                                                           \
  "fmax v12.4s, v12.4s, %[vzero].4s \n" /*relu*/                           \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v14.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v13.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v14.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v13.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "ext  v16.16b, v8.16b, v9.16b, #4 \n"                  /* v16 = 1234*/   \
  "ext  v17.16b, v8.16b, v9.16b, #8 \n" /* v16 = 2345 */ /* r3 */          \
  "fmla v15.4s ,  v8.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
  "fmla v14.4s ,  v8.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/   \
                                                                           \
  "st1 {v12.4s}, [%[doutr0]], #16     \n"                                  \
  "fmax v13.4s, v13.4s, %[vzero].4s \n" /*relu*/                           \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
  "fmla v14.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
  "fmla v14.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "ext  v16.16b, v10.16b, v11.16b, #4 \n" /* v16 = 1234*/                  \
  "ext  v17.16b, v10.16b, v11.16b, #8 \n" /* v16 = 2345 */                 \
                                                                           \
  "st1 {v13.4s}, [%[doutr1]], #16     \n" /* r3 */                         \
  "fmla v15.4s ,  v10.4s,  %[w2].s[0]\n"  /* outr00 += din0_0123 * w0[0]*/ \
                                                                           \
  "fmax v14.4s, v14.4s, %[vzero].4s \n" /*relu*/                           \
                                                                           \
  "fmla v15.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/  \
                                                                           \
  "fmla v15.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/  \
                                                                           \
  "st1 {v14.4s}, [%[doutr2]], #16     \n"                                  \
                                                                           \
  "fmax v15.4s, v15.4s, %[vzero].4s \n" /*relu*/                           \
                                                                           \
  "st1 {v15.4s}, [%[doutr3]], #16     \n"

#define COMPUTE_S_S1                       \
  "prfm pldl1keep, [%[din0]]\n"            \
  "prfm pldl1keep, [%[din1]]\n"            \
  "prfm pldl1keep, [%[din2]]\n"            \
  "prfm pldl1keep, [%[din3]]\n"            \
                                           \
  "ld1 {v0.4s}, [%[din0]], #16\n"          \
  "ld1 {v1.4s}, [%[din1]], #16\n"          \
  "ld1 {v2.4s}, [%[din2]], #16\n"          \
  "ld1 {v3.4s}, [%[din3]], #16\n"          \
                                           \
  "bif v0.16b, %[zero].16b, %[mask].16b\n" \
  "bif v1.16b, %[zero].16b, %[mask].16b\n" \
  "bif v2.16b, %[zero].16b, %[mask].16b\n" \
  "bif v3.16b, %[zero].16b, %[mask].16b\n" \
                                           \
  "ext v4.16b, %[zero].16b, v0.16b, #12\n" \
  "ext v5.16b, %[zero].16b, v1.16b, #12\n" \
  "ext v6.16b, %[zero].16b, v2.16b, #12\n" \
  "ext v7.16b, %[zero].16b, v3.16b, #12\n" \
                                           \
  "ext v8.16b, v0.16b, %[zero].16b, #4\n"  \
  "ext v9.16b, v1.16b, %[zero].16b, #4\n"  \
  "ext v10.16b, v2.16b, %[zero].16b, #4\n" \
  "ext v11.16b, v3.16b, %[zero].16b, #4\n" \
                                           \
  "fmul v12.4s, v0.4s, %[wr0].s[1]\n"      \
  "fmul v13.4s, v1.4s, %[wr0].s[1]\n"      \
                                           \
  "fmul v14.4s, v1.4s, %[wr1].s[1]\n"      \
  "fmul v15.4s, v2.4s, %[wr1].s[1]\n"      \
                                           \
  "fmul v16.4s, v2.4s, %[wr2].s[1]\n"      \
  "fmul v17.4s, v3.4s, %[wr2].s[1]\n"      \
                                           \
  "fmla v12.4s, v4.4s, %[wr0].s[0]\n"      \
  "fmla v13.4s, v5.4s, %[wr0].s[0]\n"      \
                                           \
  "fmla v14.4s, v5.4s, %[wr1].s[0]\n"      \
  "fmla v15.4s, v6.4s, %[wr1].s[0]\n"      \
                                           \
  "fmla v16.4s, v6.4s, %[wr2].s[0]\n"      \
  "fmla v17.4s, v7.4s, %[wr2].s[0]\n"      \
                                           \
  "fmla v12.4s, v8.4s, %[wr0].s[2]\n"      \
  "fmla v13.4s, v9.4s, %[wr0].s[2]\n"      \
                                           \
  "fmla v14.4s, v9.4s, %[wr1].s[2]\n"      \
  "fmla v15.4s, v10.4s, %[wr1].s[2]\n"     \
                                           \
  "fmla v16.4s, v10.4s, %[wr2].s[2]\n"     \
  "fmla v17.4s, v11.4s, %[wr2].s[2]\n"     \
                                           \
  "fadd v12.4s, v12.4s, v14.4s\n"          \
  "fadd v12.4s, v12.4s, v16.4s\n"          \
                                           \
  "fadd v13.4s, v13.4s, v15.4s\n"          \
  "fadd v13.4s, v13.4s, v17.4s\n"          \
                                           \
  "fadd v12.4s, v12.4s, %[bias].4s\n"      \
  "fadd v13.4s, v13.4s, %[bias].4s\n"

#define RESULT_S_S1             \
  "prfm pldl1keep, [%[out1]]\n" \
  "prfm pldl1keep, [%[out2]]\n" \
                                \
  "st1 {v12.4s}, [%[out1]]\n"   \
  "st1 {v13.4s}, [%[out2]]\n"

#define RESULT_S_S1_RELU              \
  "prfm pldl1keep, [%[out1]]\n"       \
  "prfm pldl1keep, [%[out2]]\n"       \
                                      \
  "fmax v12.4s, v12.4s, %[zero].4s\n" \
  "fmax v13.4s, v13.4s, %[zero].4s\n" \
                                      \
  "st1 {v12.4s}, [%[out1]]\n"         \
  "st1 {v13.4s}, [%[out2]]\n"

#define COMPUTE_S_S1_P0                                   \
  "prfm pldl1keep, [%[din0]]\n"                           \
  "prfm pldl1keep, [%[din1]]\n"                           \
  "prfm pldl1keep, [%[din2]]\n"                           \
  "prfm pldl1keep, [%[din3]]\n"                           \
                                                          \
  "ld1 {v0.4s, v1.4s}, [%[din0]]\n"                       \
  "ld1 {v2.4s, v3.4s}, [%[din1]]\n"                       \
  "ld1 {v4.4s, v5.4s}, [%[din2]]\n"                       \
  "ld1 {v6.4s, v7.4s}, [%[din3]]\n"                       \
                                                          \
  "bif v0.16b, %[zero].16b, %[mask1].16b\n"               \
  "bif v1.16b, %[zero].16b, %[mask2].16b\n"               \
                                                          \
  "bif v2.16b, %[zero].16b, %[mask1].16b\n"               \
  "bif v3.16b, %[zero].16b, %[mask2].16b\n"               \
                                                          \
  "bif v4.16b, %[zero].16b, %[mask1].16b\n"               \
  "bif v5.16b, %[zero].16b, %[mask2].16b\n"               \
                                                          \
  "bif v6.16b, %[zero].16b, %[mask1].16b\n"               \
  "bif v7.16b, %[zero].16b, %[mask2].16b\n"               \
                                                          \
  "ext v8.16b, v0.16b, v1.16b, #4\n"                      \
  "ext v9.16b, v0.16b, v1.16b, #8\n"                      \
                                                          \
  "and  v12.16b, %[vbias].16b, %[vbias].16b  \n"          \
  "and  v13.16b, %[vbias].16b, %[vbias].16b  \n" /* r0 */ \
  "fmul v10.4s, v0.4s, %[wr0].s[0]\n"                     \
  "fmul v11.4s, v8.4s, %[wr0].s[1]\n"                     \
  "fmla v12.4s, v9.4s, %[wr0].s[2]\n"                     \
                                                          \
  "ext v8.16b, v2.16b, v3.16b, #4\n"                      \
  "ext v9.16b, v2.16b, v3.16b, #8\n" /* r1 */             \
  "fmul v14.4s, v2.4s, %[wr0].s[0]\n"                     \
  "fmla v10.4s, v2.4s, %[wr1].s[0]\n"                     \
                                                          \
  "fmul v15.4s, v8.4s, %[wr0].s[1]\n"                     \
  "fmla v11.4s, v8.4s, %[wr1].s[1]\n"                     \
                                                          \
  "fmla v13.4s, v9.4s, %[wr0].s[2]\n"                     \
  "fmla v12.4s, v9.4s, %[wr1].s[2]\n"                     \
                                                          \
  "ext v8.16b, v4.16b, v5.16b, #4\n"                      \
  "ext v9.16b, v4.16b, v5.16b, #8\n" /* r2 */             \
  "fmla v14.4s, v4.4s, %[wr1].s[0]\n"                     \
  "fmla v10.4s, v4.4s, %[wr2].s[0]\n"                     \
                                                          \
  "fmla v15.4s, v8.4s, %[wr1].s[1]\n"                     \
  "fmla v11.4s, v8.4s, %[wr2].s[1]\n"                     \
                                                          \
  "fmla v13.4s, v9.4s, %[wr1].s[2]\n"                     \
  "fmla v12.4s, v9.4s, %[wr2].s[2]\n"                     \
                                                          \
  "ext v8.16b, v6.16b, v7.16b, #4\n"                      \
  "ext v9.16b, v6.16b, v7.16b, #8\n"                      \
                                                          \
  "fmla v14.4s, v6.4s, %[wr2].s[0]\n"                     \
                                                          \
  "fmla v15.4s, v8.4s, %[wr2].s[1]\n"                     \
                                                          \
  "fadd v12.4s, v12.4s, v10.4s\n"                         \
                                                          \
  "fmla v13.4s, v9.4s, %[wr2].s[2]\n"                     \
                                                          \
  "fadd v12.4s, v12.4s, v11.4s\n"                         \
  "fadd v13.4s, v13.4s, v14.4s\n"                         \
  "fadd v13.4s, v13.4s, v15.4s\n"  // \
                    // "prfm pldl1keep, [%[out1]]\n" \
                    // "prfm pldl1keep, [%[out2]]\n" \
                    // \
                    // "st1 {v12.4s}, [%[out1]]\n" \
                    // "st1 {v13.4s}, [%[out2]]\n" \

#define PARAM1                                                             \
  [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1),   \
      [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3),                \
      [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5),                \
      [doutr0] "+r"(doutr0), [doutr1] "+r"(doutr1), [doutr2] "+r"(doutr2), \
      [doutr3] "+r"(doutr3)

#define PARAM2                                                        \
  [w0] "w"(wr0), [w1] "w"(wr1), [w2] "w"(wr2), [bias_val] "r"(vbias), \
      [vmask] "r"(vmask), [vzero] "w"(vzero),                         \
      [right_pad_num] "r"(right_pad_num)

#define ASM_PARAM                                                             \
  "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", \
      "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",   \
      "v20", "v21", "v22", "v23", "v24", "v25"

#else
#define INIT_S1                                                    \
  "pld [%[din0_ptr]]                             @ preload data\n" \
  "pld [%[din1_ptr]]                      @ preload data\n"        \
  "pld [%[din2_ptr]]                      @ preload data\n"        \
  "pld [%[din3_ptr]]                      @ preload data\n"        \
                                                                   \
  "vld1.32  {d16-d18}, [%[din0_ptr]]!    @ load din r0\n"          \
  "vld1.32  {d20-d22}, [%[din1_ptr]]!    @ load din r1\n"          \
  "vld1.32  {d24-d26}, [%[din2_ptr]]!    @ load din r2\n"          \
  "vld1.32  {d28-d30}, [%[din3_ptr]]!    @ load din r3\n"          \
                                                                   \
  "vdup.32 q4, %[bias_val]                            @ and \n"    \
  "vdup.32 q5, %[bias_val]                            @ and \n"

#define LEFT_COMPUTE_S1                                            \
  "vext.32  q6, %q[vzero], q8, #3     @ 0012\n"                    \
  "vext.32  q7, q8, q9, #1     @ 1234\n" /* r0 */                  \
  "vmla.f32 q4, q8, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"           \
                                                                   \
  "sub %[din0_ptr], #12 @ 1pad + 2 float data overlap\n"           \
  "sub %[din1_ptr], #12 @ 1pad + 2 float data overlap\n"           \
  "sub %[din2_ptr], #12 @ 1pad + 2 float data overlap\n"           \
  "sub %[din3_ptr], #12 @ 1pad + 2 float data overlap\n"           \
                                                                   \
  "vmla.f32 q4, q6, %e[wr0][0]  @ q4 += 1234 * wr0[0]\n"           \
                                                                   \
  "pld [%[din0_ptr]]                             @ preload data\n" \
  "pld [%[din1_ptr]]                             @ preload data\n" \
  "pld [%[din2_ptr]]                             @ preload data\n" \
  "pld [%[din3_ptr]]                             @ preload data\n" \
                                                                   \
  "vmla.f32 q4, q7, %f[wr0][0]  @ q4 += 1234 * wr0[2]\n"           \
                                                                   \
  "vext.32  q6, %q[vzero], q10, #3     @ 0012\n"                   \
  "vext.32  q7, q10, q11, #1     @ 1234\n"                         \
                                                                   \
  /* r1 */                                                         \
  "vmla.f32 q5, q10, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"          \
  "vmla.f32 q4, q10, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"          \
                                                                   \
  "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0\n"          \
  "vld1.32  {d20-d21}, [%[din1_ptr]]!    @ load din r0\n"          \
                                                                   \
  "vmla.f32 q5, q6, %e[wr0][0]  @ q4 += 1234 * wr0[0]\n"           \
  "vmla.f32 q4, q6, %e[wr1][0]  @ q4 += 1234 * wr0[0]\n"           \
                                                                   \
  "vld1.32  {d18}, [%[din0_ptr]]    @ load din r0\n"               \
  "vld1.32  {d22}, [%[din1_ptr]]    @ load din r0\n"               \
                                                                   \
  "vmla.f32 q5, q7, %f[wr0][0]  @ q4 += 1234 * wr0[2]\n"           \
  "vmla.f32 q4, q7, %f[wr1][0]  @ q4 += 1234 * wr0[2]\n"           \
                                                                   \
  "vext.32  q6, %q[vzero], q12, #3     @ 0012\n"                   \
  "vext.32  q7, q12, q13, #1     @ 1234\n"                         \
                                                                   \
  /* r2 */                                                         \
  "vmla.f32 q5, q12, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"          \
  "vmla.f32 q4, q12, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"          \
                                                                   \
  "vld1.32  {d24-d25}, [%[din2_ptr]]!    @ load din r0\n"          \
                                                                   \
  "vmla.f32 q5, q6, %e[wr1][0]  @ q4 += 1234 * wr0[0]\n"           \
  "vmla.f32 q4, q6, %e[wr2][0]  @ q4 += 1234 * wr0[0]\n"           \
                                                                   \
  "vld1.32  {d26}, [%[din2_ptr]]    @ load din r0\n"               \
                                                                   \
  "vmla.f32 q5, q7, %f[wr1][0]  @ q4 += 1234 * wr0[2]\n"           \
  "vmla.f32 q4, q7, %f[wr2][0]  @ q4 += 1234 * wr0[2]\n"           \
                                                                   \
  "vext.32  q6, %q[vzero], q14, #3     @ 0012\n"                   \
  "vext.32  q7, q14, q15, #1     @ 1234\n"

#define LEFT_RESULT_S1                                                        \
  /* r3 */                                                                    \
  "vmla.f32 q5, q14, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                              \
  "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r0\n"                     \
  "vst1.32  {d8-d9},   [%[dout_ptr1]]!  @ store result, add pointer\n"        \
                                                                              \
  "vmla.f32 q5, q6, %e[wr2][0]  @ q4 += 1234 * wr0[0]\n"                      \
                                                                              \
  "vld1.32  {d30}, [%[din3_ptr]]    @ load din r0\n"                          \
  "vdup.32 q4, %[bias_val]                            @ and \n"               \
                                                                              \
  "vmla.f32 q5, q7, %f[wr2][0]  @ q4 += 1234 * wr0[2]\n"                      \
                                                                              \
  "vext.32  q6, q8, q9, #1     @ 1234\n"                                      \
  "vext.32  q7, q8, q9, #2     @ 2345\n"                                      \
  "cmp %[cnt], #1                             @ check whether has mid cols\n" \
                                                                              \
  "vst1.32  {d10-d11},   [%[dout_ptr2]]!  @ store result, add pointer\n"      \
                                                                              \
  "vdup.32 q5, %[bias_val]                            @ and \n"               \
  "blt  3f                                @ jump to main loop start point\n"

#define MID_COMPUTE_S1                                                 \
  "1:                                    @ right pad entry\n" /* r0 */ \
  "vmla.f32 q4, q8, %e[wr0][0]  @ q4 += 0123 * wr0[0]\n"               \
                                                                       \
  "pld [%[din0_ptr]]                             @ preload data\n"     \
  "pld [%[din1_ptr]]                             @ preload data\n"     \
  "pld [%[din2_ptr]]                             @ preload data\n"     \
  "pld [%[din3_ptr]]                             @ preload data\n"     \
                                                                       \
  "vmla.f32 q4, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"               \
                                                                       \
  "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0\n"              \
                                                                       \
  "vmla.f32 q4, q7, %f[wr0][0]  @ q4 += 2345 * wr0[2]\n"               \
                                                                       \
  "vld1.32  {d18}, [%[din0_ptr]]    @ load din r0\n"                   \
                                                                       \
  "vext.32  q6, q10, q11, #1     @ 1234\n"                             \
  "vext.32  q7, q10, q11, #2     @ 2345\n" /* r1 */                    \
  "vmla.f32 q5, q10, %e[wr0][0]  @ q4 += 1234 * wr0[0]\n"              \
  "vmla.f32 q4, q10, %e[wr1][0]  @ q4 += 1234 * wr0[1]\n"              \
                                                                       \
  "vld1.32  {d20-d21}, [%[din1_ptr]]!    @ load din r0\n"              \
                                                                       \
  "vmla.f32 q5, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"               \
  "vmla.f32 q4, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"               \
                                                                       \
  "vld1.32  {d22}, [%[din1_ptr]]    @ load din r0\n"                   \
                                                                       \
  "vmla.f32 q5, q7, %f[wr0][0]  @ q4 += 1234 * wr0[1]\n"               \
  "vmla.f32 q4, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"               \
                                                                       \
  "vext.32  q6, q12, q13, #1     @ 1234\n"                             \
  "vext.32  q7, q12, q13, #2     @ 2345\n" /* r2 */                    \
  "vmla.f32 q5, q12, %e[wr1][0]  @ q4 += 1234 * wr0[0]\n"              \
  "vmla.f32 q4, q12, %e[wr2][0]  @ q4 += 1234 * wr0[1]\n"              \
                                                                       \
  "vld1.32  {d24-d25}, [%[din2_ptr]]!    @ load din r0\n"              \
                                                                       \
  "vmla.f32 q5, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"               \
  "vmla.f32 q4, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"               \
                                                                       \
  "vld1.32  {d26}, [%[din2_ptr]]    @ load din r0\n"                   \
                                                                       \
  "vmla.f32 q5, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"               \
  "vmla.f32 q4, q7, %f[wr2][0]  @ q4 += 1234 * wr0[1]\n"               \
                                                                       \
  "vext.32  q6, q14, q15, #1     @ 1234\n"                             \
  "vext.32  q7, q14, q15, #2     @ 2345\n"

#define MID_RESULT_S1                                                    \
  /* r3 */                                                               \
  "vmla.f32 q5, q14, %e[wr2][0]  @ q4 += 0123 * wr0[0]\n"                \
                                                                         \
  "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r0\n"                \
  "vst1.32  {d8-d9},   [%[dout_ptr1]]!  @ store result, add pointer\n"   \
                                                                         \
  "vmla.f32 q5, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"                 \
                                                                         \
  "vld1.32  {d30}, [%[din3_ptr]]    @ load din r0\n"                     \
  "vdup.32 q4, %[bias_val]                            @ and \n"          \
                                                                         \
  "vmla.f32 q5, q7, %f[wr2][0]  @ q4 += 2345 * wr0[2]\n"                 \
                                                                         \
  "vext.32  q6, q8, q9, #1     @ 1234\n"                                 \
  "vext.32  q7, q8, q9, #2     @ 2345\n"                                 \
                                                                         \
  "vst1.32  {d10-d11},   [%[dout_ptr2]]!  @ store result, add pointer\n" \
                                                                         \
  "subs %[cnt], #1 @ loop count minus 1\n"                               \
                                                                         \
  "vdup.32 q5, %[bias_val]                            @ and \n"          \
                                                                         \
  "bne    1b                             @ jump to main loop start point\n"

#define RIGHT_COMPUTE_S1                                                     \
  "3:                                    @ right pad entry\n"                \
                                                                             \
  "subs %[din0_ptr], #16 \n"                                                 \
  "subs %[din1_ptr], #16 \n"                                                 \
  "subs %[din2_ptr], #16 \n"                                                 \
  "subs %[din3_ptr], #16 \n"                                                 \
  "subs %[din0_ptr], %[right_pad_num] \n"                                    \
  "subs %[din1_ptr], %[right_pad_num] \n"                                    \
  "subs %[din2_ptr], %[right_pad_num] \n"                                    \
  "subs %[din3_ptr], %[right_pad_num] \n"                                    \
  "subs %[dout_ptr1], %[right_pad_num] \n"                                   \
  "subs %[dout_ptr2], %[right_pad_num] \n"                                   \
                                                                             \
  "vld1.32  {d16-d18}, [%[din0_ptr]]    \n"                                  \
  "vld1.32  {d20-d22}, [%[din1_ptr]]   \n"                                   \
  "vld1.32  {d24-d26}, [%[din2_ptr]]    \n"                                  \
  "vld1.32  {d28-d30}, [%[din3_ptr]]    \n"                                  \
                                                                             \
  "vld1.32  {d27}, [%[vmask]]!    @ load din r0\n"                           \
                                                                             \
  "vbif d18, %e[vzero], d27             @ bit select, deal with right pad\n" \
                                                                             \
  "vbif d22, %e[vzero], d27             @ bit select, deal with right pad\n" \
                                                                             \
  "vext.32  q6, q8, q9, #1     @ 1234\n"                                     \
  "vext.32  q7, q8, q9, #2     @ 2345\n" /* r0 */                            \
  "vmla.f32 q4, q8, %e[wr0][0]  @ q4 += 0123 * wr0[0]\n"                     \
                                                                             \
  "vbif d26, %e[vzero], d27             @ bit select, deal with right pad\n" \
                                                                             \
  "vmla.f32 q4, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                             \
  "vbif d30, %e[vzero], d27             @ bit select, deal with right pad\n" \
                                                                             \
  "vmla.f32 q4, q7, %f[wr0][0]  @ q4 += 2345 * wr0[2]\n"                     \
                                                                             \
  "vext.32  q6, q10, q11, #1     @ 1234\n"                                   \
  "vext.32  q7, q10, q11, #2     @ 2345\n" /* r1 */                          \
  "vmla.f32 q5, q10, %e[wr0][0]  @ q4 += 1234 * wr0[0]\n"                    \
  "vmla.f32 q4, q10, %e[wr1][0]  @ q4 += 1234 * wr0[1]\n"                    \
                                                                             \
  "vmla.f32 q5, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"                     \
  "vmla.f32 q4, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                             \
  "vld1.32  {d16-d17}, [%[dout_ptr1]]    @ load din r0\n"                    \
  "vld1.32  {d20-d21}, [%[dout_ptr2]]    @ load din r0\n"                    \
                                                                             \
  "vmla.f32 q5, q7, %f[wr0][0]  @ q4 += 1234 * wr0[1]\n"                     \
  "vmla.f32 q4, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                             \
  "vext.32  q6, q12, q13, #1     @ 1234\n"                                   \
  "vext.32  q7, q12, q13, #2     @ 2345\n" /* r2 */                          \
  "vmla.f32 q5, q12, %e[wr1][0]  @ q4 += 1234 * wr0[0]\n"                    \
  "vmla.f32 q4, q12, %e[wr2][0]  @ q4 += 1234 * wr0[1]\n"                    \
                                                                             \
  "vmla.f32 q5, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"                     \
  "vmla.f32 q4, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                             \
  "vmla.f32 q5, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"                     \
  "vmla.f32 q4, q7, %f[wr2][0]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                             \
  "vext.32  q6, q14, q15, #1     @ 1234\n"                                   \
  "vext.32  q7, q14, q15, #2     @ 2345\n"

#define RIGHT_RESULT_S1                                                \
  /* r3 */                                                             \
  "vmla.f32 q5, q14, %e[wr2][0]  @ q4 += 0123 * wr0[0]\n"              \
                                                                       \
  "vmla.f32 q5, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"               \
                                                                       \
  "vst1.32  {d8-d9},   [%[dout_ptr1]]!  @ store result, add pointer\n" \
                                                                       \
  "vmla.f32 q5, q7, %f[wr2][0]  @ q4 += 2345 * wr0[2]\n"               \
                                                                       \
  "vst1.32  {d10-d11},   [%[dout_ptr2]]!  @ store result, add pointer\n"

#define LEFT_RESULT_S1_RELU                                                   \
  /* r3 */                                                                    \
  "vmla.f32 q5, q14, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                              \
  "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r0\n"                     \
  "vmax.f32  q4, q4, %q[vzero]  @ relu \n"                                    \
                                                                              \
  "vmla.f32 q5, q6, %e[wr2][0]  @ q4 += 1234 * wr0[0]\n"                      \
                                                                              \
  "vld1.32  {d30}, [%[din3_ptr]]    @ load din r0\n"                          \
  "vst1.32  {d8-d9},   [%[dout_ptr1]]!  @ store result, add pointer\n"        \
                                                                              \
  "vmla.f32 q5, q7, %f[wr2][0]  @ q4 += 1234 * wr0[2]\n"                      \
                                                                              \
  "vext.32  q6, q8, q9, #1     @ 1234\n"                                      \
  "vext.32  q7, q8, q9, #2     @ 2345\n"                                      \
  "vdup.32 q4, %[bias_val]                            @ and \n"               \
                                                                              \
  "vmax.f32  q5, q5, %q[vzero]  @ relu \n"                                    \
                                                                              \
  "cmp %[cnt], #1                             @ check whether has mid cols\n" \
                                                                              \
  "vst1.32  {d10-d11},   [%[dout_ptr2]]!  @ store result, add pointer\n"      \
                                                                              \
  "vdup.32 q5, %[bias_val]                            @ and \n"               \
  "blt  3f                                @ jump to main loop start point\n"

#define MID_RESULT_S1_RELU                                               \
  /* r3 */                                                               \
  "vmla.f32 q5, q14, %e[wr2][0]  @ q4 += 0123 * wr0[0]\n"                \
                                                                         \
  "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r0\n"                \
  "vmax.f32  q4, q4, %q[vzero]  @ relu \n"                               \
                                                                         \
  "vmla.f32 q5, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"                 \
                                                                         \
  "vld1.32  {d30}, [%[din3_ptr]]    @ load din r0\n"                     \
  "vst1.32  {d8-d9},   [%[dout_ptr1]]!  @ store result, add pointer\n"   \
                                                                         \
  "vmla.f32 q5, q7, %f[wr2][0]  @ q4 += 2345 * wr0[2]\n"                 \
                                                                         \
  "vext.32  q6, q8, q9, #1     @ 1234\n"                                 \
  "vext.32  q7, q8, q9, #2     @ 2345\n"                                 \
  "vdup.32 q4, %[bias_val]                            @ and \n"          \
                                                                         \
  "vmax.f32  q5, q5, %q[vzero]  @ relu \n"                               \
                                                                         \
  "vst1.32  {d10-d11},   [%[dout_ptr2]]!  @ store result, add pointer\n" \
                                                                         \
  "subs %[cnt], #1 @ loop count minus 1\n"                               \
                                                                         \
  "vdup.32 q5, %[bias_val]                            @ and \n"          \
                                                                         \
  "bne    1b                             @ jump to main loop start point\n"

#define RIGHT_RESULT_S1_RELU                                           \
  /* r3 */                                                             \
  "vmla.f32 q5, q14, %e[wr2][0]  @ q4 += 0123 * wr0[0]\n"              \
                                                                       \
  "vmax.f32  q4, q4, %q[vzero]  @ relu \n"                             \
                                                                       \
  "vmla.f32 q5, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"               \
                                                                       \
  "vmla.f32 q5, q7, %f[wr2][0]  @ q4 += 2345 * wr0[2]\n"               \
  "vst1.32  {d8-d9},   [%[dout_ptr1]]!  @ store result, add pointer\n" \
                                                                       \
  "vmax.f32  q5, q5, %q[vzero]  @ relu \n"                             \
                                                                       \
  "vst1.32  {d10-d11},   [%[dout_ptr2]]!  @ store result, add pointer\n"

#define COMPUTE_S_S1                 \
  "pld [%[din0]]\n"                  \
  "pld [%[din1]]\n"                  \
  "pld [%[din2]]\n"                  \
  "pld [%[din3]]\n"                  \
                                     \
  "vld1.32 {d12-d13}, [%[din0]]!\n"  \
  "vld1.32 {d14-d15}, [%[din1]]!\n"  \
  "vld1.32 {d16-d17}, [%[din2]]!\n"  \
  "vld1.32 {d18-d19}, [%[din3]]!\n"  \
                                     \
  "vbif q6, %q[vzero], %q[mask]\n"   \
  "vbif q7, %q[vzero], %q[mask]\n"   \
  "vbif q8, %q[vzero], %q[mask]\n"   \
  "vbif q9, %q[vzero], %q[mask]\n"   \
                                     \
  "vmul.f32 q14, q6, %e[wr0][1]\n"   \
  "vmul.f32 q15, q7, %e[wr0][1]\n"   \
                                     \
  "vmla.f32 q14, q7, %e[wr1][1]\n"   \
  "vmla.f32 q15, q8, %e[wr1][1]\n"   \
                                     \
  "vmla.f32 q14, q8, %e[wr2][1]\n"   \
  "vmla.f32 q15, q9, %e[wr2][1]\n"   \
                                     \
  "vext.32 q10, %q[vzero], q6, #3\n" \
  "vext.32 q11, %q[vzero], q7, #3\n" \
  "vext.32 q12, %q[vzero], q8, #3\n" \
  "vext.32 q13, %q[vzero], q9, #3\n" \
                                     \
  "vmla.f32 q14, q10, %e[wr0][0]\n"  \
  "vmla.f32 q15, q11, %e[wr0][0]\n"  \
                                     \
  "vmla.f32 q14, q11, %e[wr1][0]\n"  \
  "vmla.f32 q15, q12, %e[wr1][0]\n"  \
                                     \
  "vmla.f32 q14, q12, %e[wr2][0]\n"  \
  "vmla.f32 q15, q13, %e[wr2][0]\n"  \
                                     \
  "vext.32 q10, q6, %q[vzero], #1\n" \
  "vext.32 q11, q7, %q[vzero], #1\n" \
  "vext.32 q12, q8, %q[vzero], #1\n" \
  "vext.32 q13, q9, %q[vzero], #1\n" \
                                     \
  "vmla.f32 q14, q10, %f[wr0][0]\n"  \
  "vmla.f32 q15, q11, %f[wr0][0]\n"  \
                                     \
  "vmla.f32 q14, q11, %f[wr1][0]\n"  \
  "vmla.f32 q15, q12, %f[wr1][0]\n"  \
                                     \
  "vmla.f32 q14, q12, %f[wr2][0]\n"  \
  "vmla.f32 q15, q13, %f[wr2][0]\n"  \
                                     \
  "vadd.f32 q14, q14, %q[bias]\n"    \
  "vadd.f32 q15, q15, %q[bias]\n"

#define RESULT_S_S1                \
  "pld [%[out1]]\n"                \
  "pld [%[out2]]\n"                \
                                   \
  "vst1.32 {d28-d29}, [%[out1]]\n" \
  "vst1.32 {d30-d31}, [%[out2]]\n"

#define RESULT_S_S1_RELU           \
  "pld [%[out1]]\n"                \
  "pld [%[out2]]\n"                \
                                   \
  "vmax.f32 q14, q14, %q[vzero]\n" \
  "vmax.f32 q15, q15, %q[vzero]\n" \
                                   \
  "vst1.32 {d28-d29}, [%[out1]]\n" \
  "vst1.32 {d30-d31}, [%[out2]]\n"

#define COMPUTE_S_S1_P0                                                       \
  "pld [%[din0]]\n"                                                           \
  "pld [%[din1]]\n"                                                           \
  "pld [%[din2]]\n"                                                           \
  "pld [%[din3]]\n"                                                           \
  "vld1.32  {d16-d18}, [%[din0]]    @ load din r0\n"                          \
  "vld1.32  {d20-d22}, [%[din1]]    @ load din r1\n"                          \
  "vld1.32  {d24-d26}, [%[din2]]    @ load din r2\n"                          \
  "vld1.32  {d28-d30}, [%[din3]]    @ load din r3\n"                          \
                                                                              \
  "vdup.32 q4, %[bias_val]                            @ and \n"               \
  "vdup.32 q5, %[bias_val]                            @ and \n"               \
                                                                              \
  "vld1.32  {d19}, [%[vmask]]!    @ load din r0\n"                            \
  "vld1.32  {d23}, [%[vmask]]!    @ load din r0\n"                            \
                                                                              \
  "vld1.32  {d27}, [%[vmask]]!    @ load din r0\n"                            \
                                                                              \
  "vbif d16, %e[vzero], d19              @ bit select, deal with right pad\n" \
  "vbif d20, %e[vzero], d19              @ bit select, deal with right pad\n" \
                                                                              \
  "vbif d17, %e[vzero], d23              @ bit select, deal with right pad\n" \
  "vbif d21, %e[vzero], d23              @ bit select, deal with right pad\n" \
                                                                              \
  "vbif d18, %e[vzero], d27             @ bit select, deal with right pad\n"  \
  "vbif d22, %e[vzero], d27             @ bit select, deal with right pad\n"  \
                                                                              \
  "vext.32  q6, q8, q9, #1     @ 1234\n"                                      \
  "vext.32  q7, q8, q9, #2     @ 2345\n" /* r0 */                             \
  "vmla.f32 q4, q8, %e[wr0][0]  @ q4 += 0123 * wr0[0]\n"                      \
                                                                              \
  "vbif d24, %e[vzero], d19              @ bit select, deal with right pad\n" \
  "vbif d25, %e[vzero], d23              @ bit select, deal with right pad\n" \
  "vbif d26, %e[vzero], d27             @ bit select, deal with right pad\n"  \
                                                                              \
  "vmla.f32 q4, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"                      \
                                                                              \
  "vbif d28, %e[vzero], d19              @ bit select, deal with right pad\n" \
  "vbif d29, %e[vzero], d23              @ bit select, deal with right pad\n" \
  "vbif d30, %e[vzero], d27             @ bit select, deal with right pad\n"  \
                                                                              \
  "vmla.f32 q4, q7, %f[wr0][0]  @ q4 += 2345 * wr0[2]\n"                      \
                                                                              \
  "vext.32  q6, q10, q11, #1     @ 1234\n"                                    \
  "vext.32  q7, q10, q11, #2     @ 2345\n" /* r1 */                           \
  "vmla.f32 q5, q10, %e[wr0][0]  @ q4 += 1234 * wr0[0]\n"                     \
  "vmla.f32 q4, q10, %e[wr1][0]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                              \
  "vmul.f32 q8, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"                      \
  "vmul.f32 q10, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                              \
  "vmul.f32 q9, q7, %f[wr0][0]  @ q4 += 1234 * wr0[1]\n"                      \
  "vmul.f32 q11, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                              \
  "vext.32  q6, q12, q13, #1     @ 1234\n"                                    \
  "vext.32  q7, q12, q13, #2     @ 2345\n" /* r2 */                           \
  "vmla.f32 q5, q12, %e[wr1][0]  @ q4 += 1234 * wr0[0]\n"                     \
  "vmla.f32 q4, q12, %e[wr2][0]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                              \
  "vmla.f32 q8, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"                      \
  "vmla.f32 q10, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                              \
  "vmla.f32 q9, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"                      \
  "vmla.f32 q11, q7, %f[wr2][0]  @ q4 += 1234 * wr0[1]\n"                     \
                                                                              \
  "vext.32  q6, q14, q15, #1     @ 1234\n"                                    \
  "vext.32  q7, q14, q15, #2     @ 2345\n" /* r3 */                           \
  "vmla.f32 q5, q14, %e[wr2][0]  @ q4 += 0123 * wr0[0]\n"                     \
                                                                              \
  "vmla.f32 q8, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"                      \
  "vadd.f32 q4, q4, q10         @ q4 += q10 \n"                               \
                                                                              \
  "pld [%[out1]]\n"                                                           \
  "pld [%[out2]]\n"                                                           \
                                                                              \
  "vmla.f32 q9, q7, %f[wr2][0]  @ q4 += 2345 * wr0[2]\n"                      \
  "vadd.f32 q14, q4, q11         @ q4 += q10 \n"                              \
                                                                              \
  "vadd.f32 q5, q5, q8         @ q4 += q10 \n"                                \
  "vadd.f32 q15, q5, q9         @ q4 += q10 \n"

#define PARAM1                                                               \
  [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1),                        \
      [din0_ptr] "+r"(din_ptr0), [din1_ptr] "+r"(din_ptr1),                  \
      [din2_ptr] "+r"(din_ptr2), [din3_ptr] "+r"(din_ptr3), [cnt] "+r"(cnt), \
      [vmask] "+r"(vmask_ptr)

#define PARAM2                                                              \
  [wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), [bias_val] "r"(bias_val), \
      [vzero] "w"(vzero), [right_pad_num] "r"(right_pad_num)

#define ASM_PARAM                                                          \
  "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", \
      "q13", "q14", "q15"

#endif

inline std::pair<uint32_t, uint32_t> right_mask_3x3s1_fp32(
    int w_in, int w_out, int left_padding, unsigned int *vmask) {
  //! for 4x6 convolution window
  const unsigned int right_pad_idx[4] = {1, 0};

  int tile_w = w_out >> 2;
  int remain = w_out % 4;
  int cnt_col;
  if (left_padding > 0)
    cnt_col = tile_w - 1;
  else
    cnt_col = tile_w;
  unsigned int size_pad_right =
      (unsigned int)((6 - left_padding) + (tile_w << 2) - w_in);

  // Determine whether the last 4 outputs are related to right padding
  if (remain == 0 && w_out + 2 > w_in + left_padding) {
    size_pad_right = 6 - size_pad_right;
    remain = 4;
    cnt_col--;
  }

  // at the beginning vmask[5,4,...,size_pad_right] = 0xffffffff！
  // Now the sliding window needs to move left 4-remain
  // so the valid data grows! i.e. size_pad_right decrease!
  size_pad_right -= (4 - remain);

  uint32x4_t vmask_rp2 =
      vcgeq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));
  vst1q_u32(vmask, vmask_rp2);
  return std::make_pair(cnt_col, remain);
}

/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias,
 * width > 4
 */
void conv_depthwise_3x3s1p1_bias_no_relu(float *dout,
                                         const float *din,
                                         const float *weights,
                                         const float *bias,
                                         bool flag_bias,
                                         bool flag_relu,
                                         const int num,
                                         const int ch_in,
                                         const int h_in,
                                         const int w_in,
                                         const int h_out,
                                         const int w_out,
                                         ARMContext *ctx) {
  //! pad is done implicit
  const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  float *zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, (w_in + 6) * sizeof(float));
  float *write_ptr = zero_ptr + (w_in + 6);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  unsigned int vmask[4];
  auto &&res = right_mask_3x3s1_fp32(w_in, w_out, 1, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num = (4 - cnt_remain) * 4;

  float32x4_t vzero = vdupq_n_f32(0.f);

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float *dout_ptr = dout_batch + c * size_out_channel;

      const float *din_ch_ptr = din_batch + c * size_in_channel;

      float bias_val = flag_bias ? bias[c] : 0.f;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};

      const float *wei_ptr = weights + c * w_stride;

      float32x4_t wr0 = vld1q_f32(wei_ptr);
      float32x4_t wr1 = vld1q_f32(wei_ptr + 3);
      float32x4_t wr2 = vld1q_f32(wei_ptr + 6);

      float *doutr0 = dout_ptr;
      float *doutr1 = doutr0 + w_out;
      float *doutr2 = doutr1 + w_out;
      float *doutr3 = doutr2 + w_out;

      const float *dr0 = din_ch_ptr;
      const float *dr1 = dr0 + w_in;
      const float *dr2 = dr1 + w_in;
      const float *dr3 = dr2 + w_in;
      const float *dr4 = dr3 + w_in;
      const float *dr5 = dr4 + w_in;

      const float *din_ptr0 = dr0;
      const float *din_ptr1 = dr1;
      const float *din_ptr2 = dr2;
      const float *din_ptr3 = dr3;
      const float *din_ptr4 = dr4;
      const float *din_ptr5 = dr5;
      float *ptr_zero = const_cast<float *>(zero);
#ifdef __aarch64__
      for (int i = 0; i < h_out; i += 4) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;
        din_ptr5 = dr5;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        doutr2 = doutr1 + w_out;
        doutr3 = doutr2 + w_out;
        if (i == 0) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          din_ptr4 = dr3;
          din_ptr5 = dr4;
          dr0 = dr3;
          dr1 = dr4;
          dr2 = dr5;
        } else {
          dr0 = dr4;
          dr1 = dr5;
          dr2 = dr1 + w_in;
        }
        dr3 = dr2 + w_in;
        dr4 = dr3 + w_in;
        dr5 = dr4 + w_in;

        //! process bottom pad
        if (i + 5 > h_in) {
          switch (i + 5 - h_in) {
            case 5:
              din_ptr1 = zero_ptr;
            case 4:
              din_ptr2 = zero_ptr;
            case 3:
              din_ptr3 = zero_ptr;
            case 2:
              din_ptr4 = zero_ptr;
            case 1:
              din_ptr5 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 4 > h_out) {
          switch (i + 4 - h_out) {
            case 3:
              doutr1 = write_ptr;
            case 2:
              doutr2 = write_ptr;
            case 1:
              doutr3 = write_ptr;
            default:
              break;
          }
        }

        int cnt = cnt_col;
        asm volatile(INIT_S1 LEFT_COMPUTE_S1 LEFT_RESULT_S1 MID_COMPUTE_S1
                         MID_RESULT_S1 RIGHT_COMPUTE_S1 RIGHT_RESULT_S1
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
        dout_ptr = dout_ptr + 4 * w_out;
      }
#else
      for (int i = 0; i < h_in; i += 2) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = dout_ptr + w_out;
        // unsigned int* rst_mask = rmask;

        if (i == 0) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          dr0 = dr1;
          dr1 = dr2;
          dr2 = dr3;
          dr3 = dr2 + w_in;
        } else {
          dr0 = dr2;
          dr1 = dr3;
          dr2 = dr1 + w_in;
          dr3 = dr2 + w_in;
        }
        //! process bottom pad
        if (i + 3 > h_in) {
          switch (i + 3 - h_in) {
            case 3:
              din_ptr1 = zero_ptr;
            case 2:
              din_ptr2 = zero_ptr;
            case 1:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;

        unsigned int *vmask_ptr = vmask;
        asm volatile(INIT_S1 LEFT_COMPUTE_S1 LEFT_RESULT_S1 MID_COMPUTE_S1
                         MID_RESULT_S1 RIGHT_COMPUTE_S1 RIGHT_RESULT_S1
                     : PARAM1
                     : PARAM2
                     : ASM_PARAM);
        dout_ptr += 2 * w_out;
      }  //! end of processing mid rows
#endif
    }
    LITE_PARALLEL_END()
  }
}

void conv_depthwise_3x3s1p1_bias_relu(float *dout,
                                      const float *din,
                                      const float *weights,
                                      const float *bias,
                                      bool flag_bias,
                                      bool flag_relu,
                                      const int num,
                                      const int ch_in,
                                      const int h_in,
                                      const int w_in,
                                      const int h_out,
                                      const int w_out,
                                      ARMContext *ctx) {
  //! pad is done implicit
  const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  float *zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, (w_in + 6) * sizeof(float));
  float *write_ptr = zero_ptr + (w_in + 6);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  unsigned int vmask[4];
  auto &&res = right_mask_3x3s1_fp32(w_in, w_out, 1, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32_t right_pad_num = (4 - cnt_remain) * 4;

  float32x4_t vzero = vdupq_n_f32(0.f);

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float *dout_ptr = dout_batch + c * size_out_channel;

      const float *din_ch_ptr = din_batch + c * size_in_channel;

      float bias_val = flag_bias ? bias[c] : 0.f;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};

      const float *wei_ptr = weights + c * w_stride;

      float32x4_t wr0 = vld1q_f32(wei_ptr);
      float32x4_t wr1 = vld1q_f32(wei_ptr + 3);
      float32x4_t wr2 = vld1q_f32(wei_ptr + 6);

      float *doutr0 = dout_ptr;
      float *doutr1 = doutr0 + w_out;
      float *doutr2 = doutr1 + w_out;
      float *doutr3 = doutr2 + w_out;

      const float *dr0 = din_ch_ptr;
      const float *dr1 = dr0 + w_in;
      const float *dr2 = dr1 + w_in;
      const float *dr3 = dr2 + w_in;
      const float *dr4 = dr3 + w_in;
      const float *dr5 = dr4 + w_in;

      const float *din_ptr0 = dr0;
      const float *din_ptr1 = dr1;
      const float *din_ptr2 = dr2;
      const float *din_ptr3 = dr3;
      const float *din_ptr4 = dr4;
      const float *din_ptr5 = dr5;
      float *ptr_zero = const_cast<float *>(zero);

#ifdef __aarch64__
      for (int i = 0; i < h_out; i += 4) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;
        din_ptr5 = dr5;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        doutr2 = doutr1 + w_out;
        doutr3 = doutr2 + w_out;
        if (i == 0) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          din_ptr4 = dr3;
          din_ptr5 = dr4;
          dr0 = dr3;
          dr1 = dr4;
          dr2 = dr5;
        } else {
          dr0 = dr4;
          dr1 = dr5;
          dr2 = dr1 + w_in;
        }
        dr3 = dr2 + w_in;
        dr4 = dr3 + w_in;
        dr5 = dr4 + w_in;

        //! process bottom pad
        if (i + 5 > h_in) {
          switch (i + 5 - h_in) {
            case 5:
              din_ptr1 = zero_ptr;
            case 4:
              din_ptr2 = zero_ptr;
            case 3:
              din_ptr3 = zero_ptr;
            case 2:
              din_ptr4 = zero_ptr;
            case 1:
              din_ptr5 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 4 > h_out) {
          switch (i + 4 - h_out) {
            case 3:
              doutr1 = write_ptr;
            case 2:
              doutr2 = write_ptr;
            case 1:
              doutr3 = write_ptr;
            default:
              break;
          }
        }

        int cnt = cnt_col;
        asm volatile(
            INIT_S1 LEFT_COMPUTE_S1 LEFT_RESULT_S1_RELU MID_COMPUTE_S1
                MID_RESULT_S1_RELU RIGHT_COMPUTE_S1 RIGHT_RESULT_S1_RELU
            : PARAM1
            : PARAM2
            : ASM_PARAM);
        dout_ptr = dout_ptr + 4 * w_out;
      }
#else
      for (int i = 0; i < h_out; i += 2) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = dout_ptr + w_out;
        // unsigned int* rst_mask = rmask;

        if (i == 0) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          dr0 = dr1;
          dr1 = dr2;
          dr2 = dr3;
          dr3 = dr2 + w_in;
        } else {
          dr0 = dr2;
          dr1 = dr3;
          dr2 = dr1 + w_in;
          dr3 = dr2 + w_in;
        }
        //! process bottom pad
        if (i + 3 > h_in) {
          switch (i + 3 - h_in) {
            case 3:
              din_ptr1 = zero_ptr;
            case 2:
              din_ptr2 = zero_ptr;
            case 1:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;

        unsigned int *vmask_ptr = vmask;
        asm volatile(
            INIT_S1 LEFT_COMPUTE_S1 LEFT_RESULT_S1_RELU MID_COMPUTE_S1
                MID_RESULT_S1_RELU RIGHT_COMPUTE_S1 RIGHT_RESULT_S1_RELU
            : PARAM1
            : PARAM2
            : ASM_PARAM);
        dout_ptr += 2 * w_out;
      }  //! end of processing mid rows
#endif
    }

    LITE_PARALLEL_END()
  }
}

/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias,
 * width <= 4
 */
void conv_depthwise_3x3s1p1_bias_s_no_relu(float *dout,
                                           const float *din,
                                           const float *weights,
                                           const float *bias,
                                           bool flag_bias,
                                           bool flag_relu,
                                           const int num,
                                           const int ch_in,
                                           const int h_in,
                                           const int w_in,
                                           const int h_out,
                                           const int w_out,
                                           ARMContext *ctx) {
  //! 3x3s1 convolution, implemented by direct algorithm
  //! pad is done implicit
  //! for 4x6 convolution window
  const int right_pad_idx[4] = {3, 2, 1, 0};
  const float zero[4] = {0.f, 0.f, 0.f, 0.f};

  float32x4_t vzero = vdupq_n_f32(0.f);
  uint32x4_t vmask_rp =
      vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(4 - w_in));
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    LITE_PARALLEL_BEGIN(i, tid, ch_in) {
      float *dout_channel = dout_batch + i * size_out_channel;
      const float *din_channel = din_batch + i * size_in_channel;
      const float *weight_ptr = weights + i * 9;
      float32x4_t wr0 = vld1q_f32(weight_ptr);
      float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
      float32x4_t wr2 = vld1q_f32(weight_ptr + 6);
      float32x4_t wbias;
      if (flag_bias) {
        wbias = vdupq_n_f32(bias[i]);
      } else {
        wbias = vdupq_n_f32(0.f);
      }

      int hs = -1;
      int he = 3;

      float out_buf1[4];
      float out_buf2[4];
      float trash_buf[4];

      int h_cnt = (h_out + 1) >> 1;
      float *doutr0 = dout_channel;
      float *doutr1 = dout_channel + w_out;

      for (int j = 0; j < h_cnt; ++j) {
        const float *dr0 = din_channel + hs * w_in;
        const float *dr1 = dr0 + w_in;
        const float *dr2 = dr1 + w_in;
        const float *dr3 = dr2 + w_in;

        if (hs == -1) {
          dr0 = zero;
        }

        switch (he - h_in) {
          case 2:
            dr2 = zero;
            doutr1 = trash_buf;
          case 1:
            dr3 = zero;
          default:
            break;
        }
#ifdef __aarch64__
        asm volatile(COMPUTE_S_S1 RESULT_S_S1
                     : [din0] "+r"(dr0),
                       [din1] "+r"(dr1),
                       [din2] "+r"(dr2),
                       [din3] "+r"(dr3)
                     : [wr0] "w"(wr0),
                       [wr1] "w"(wr1),
                       [wr2] "w"(wr2),
                       [zero] "w"(vzero),
                       [mask] "w"(vmask_rp),
                       [bias] "w"(wbias),
                       [out1] "r"(out_buf1),
                       [out2] "r"(out_buf2)
                     : "v0",
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
                       "v17");
#else
        asm volatile(COMPUTE_S_S1 RESULT_S_S1
                     : [din0] "+r"(dr0),
                       [din1] "+r"(dr1),
                       [din2] "+r"(dr2),
                       [din3] "+r"(dr3)
                     : [wr0] "w"(wr0),
                       [wr1] "w"(wr1),
                       [wr2] "w"(wr2),
                       [vzero] "w"(vzero),
                       [mask] "w"(vmask_rp),
                       [bias] "w"(wbias),
                       [out1] "r"(out_buf1),
                       [out2] "r"(out_buf2)
                     : "cc",
                       "memory",
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
        for (int w = 0; w < w_out; ++w) {
          *doutr0++ = out_buf1[w];
          *doutr1++ = out_buf2[w];
        }
        doutr0 = doutr1;
        doutr1 += w_out;
        hs += 2;
        he += 2;
      }  // end of processing heights
    }    // end of processing channels
    LITE_PARALLEL_END()
  }  // end of processing batchs
}
void conv_depthwise_3x3s1p1_bias_s_relu(float *dout,
                                        const float *din,
                                        const float *weights,
                                        const float *bias,
                                        bool flag_bias,
                                        bool flag_relu,
                                        const int num,
                                        const int ch_in,
                                        const int h_in,
                                        const int w_in,
                                        const int h_out,
                                        const int w_out,
                                        ARMContext *ctx) {
  //! 3x3s1 convolution, implemented by direct algorithm
  //! pad is done implicit
  //! for 4x6 convolution window
  const int right_pad_idx[4] = {3, 2, 1, 0};
  const float zero[4] = {0.f, 0.f, 0.f, 0.f};

  float32x4_t vzero = vdupq_n_f32(0.f);
  uint32x4_t vmask_rp =
      vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(4 - w_in));
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    LITE_PARALLEL_BEGIN(i, tid, ch_in) {
      float *dout_channel = dout_batch + i * size_out_channel;
      const float *din_channel = din_batch + i * size_in_channel;
      const float *weight_ptr = weights + i * 9;
      float32x4_t wr0 = vld1q_f32(weight_ptr);
      float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
      float32x4_t wr2 = vld1q_f32(weight_ptr + 6);
      float32x4_t wbias;
      if (flag_bias) {
        wbias = vdupq_n_f32(bias[i]);
      } else {
        wbias = vdupq_n_f32(0.f);
      }

      int hs = -1;
      int he = 3;

      float out_buf1[4];
      float out_buf2[4];
      float trash_buf[4];

      int h_cnt = (h_out + 1) >> 1;
      float *doutr0 = dout_channel;
      float *doutr1 = dout_channel + w_out;

      for (int j = 0; j < h_cnt; ++j) {
        const float *dr0 = din_channel + hs * w_in;
        const float *dr1 = dr0 + w_in;
        const float *dr2 = dr1 + w_in;
        const float *dr3 = dr2 + w_in;

        if (hs == -1) {
          dr0 = zero;
        }

        switch (he - h_in) {
          case 2:
            dr2 = zero;
            doutr1 = trash_buf;
          case 1:
            dr3 = zero;
          default:
            break;
        }
#ifdef __aarch64__
        asm volatile(COMPUTE_S_S1 RESULT_S_S1_RELU
                     : [din0] "+r"(dr0),
                       [din1] "+r"(dr1),
                       [din2] "+r"(dr2),
                       [din3] "+r"(dr3)
                     : [wr0] "w"(wr0),
                       [wr1] "w"(wr1),
                       [wr2] "w"(wr2),
                       [zero] "w"(vzero),
                       [mask] "w"(vmask_rp),
                       [bias] "w"(wbias),
                       [out1] "r"(out_buf1),
                       [out2] "r"(out_buf2)
                     : "v0",
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
                       "v17");
#else
        asm volatile(COMPUTE_S_S1 RESULT_S_S1_RELU
                     : [din0] "+r"(dr0),
                       [din1] "+r"(dr1),
                       [din2] "+r"(dr2),
                       [din3] "+r"(dr3)
                     : [wr0] "w"(wr0),
                       [wr1] "w"(wr1),
                       [wr2] "w"(wr2),
                       [vzero] "w"(vzero),
                       [mask] "w"(vmask_rp),
                       [bias] "w"(wbias),
                       [out1] "r"(out_buf1),
                       [out2] "r"(out_buf2)
                     : "cc",
                       "memory",
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
        for (int w = 0; w < w_out; ++w) {
          *doutr0++ = out_buf1[w];
          *doutr1++ = out_buf2[w];
        }
        doutr0 = doutr1;
        doutr1 += w_out;
        hs += 2;
        he += 2;
      }  // end of processing heights
    }    // end of processing channels
    LITE_PARALLEL_END()
  }  // end of processing batchs
}

/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias,
 * width > 4
 */
void conv_depthwise_3x3s1p0_bias_no_relu(float *dout,
                                         const float *din,
                                         const float *weights,
                                         const float *bias,
                                         bool flag_bias,
                                         bool flag_relu,
                                         const int num,
                                         const int ch_in,
                                         const int h_in,
                                         const int w_in,
                                         const int h_out,
                                         const int w_out,
                                         ARMContext *ctx) {
  //! pad is done implicit
  const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  float *zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, (w_in + 6) * sizeof(float));
  float *write_ptr = zero_ptr + (w_in + 6);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  unsigned int vmask[4];
  auto &&res = right_mask_3x3s1_fp32(w_in, w_out, 0, vmask);
  uint32_t cnt_col = res.first;
  uint32_t remain = res.second;
  uint32_t right_pad_num = (4 - remain) * 4;

  float32x4_t vzero = vdupq_n_f32(0.f);

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float *dout_ptr = dout_batch + c * size_out_channel;

      const float *din_ch_ptr = din_batch + c * size_in_channel;

      float bias_val = flag_bias ? bias[c] : 0.f;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};

      const float *wei_ptr = weights + c * w_stride;

      float32x4_t wr0 = vld1q_f32(wei_ptr);
      float32x4_t wr1 = vld1q_f32(wei_ptr + 3);
      float32x4_t wr2 = vld1q_f32(wei_ptr + 6);

      float *doutr0 = dout_ptr;
      float *doutr1 = doutr0 + w_out;
      float *doutr2 = doutr1 + w_out;
      float *doutr3 = doutr2 + w_out;

      const float *dr0 = din_ch_ptr;
      const float *dr1 = dr0 + w_in;
      const float *dr2 = dr1 + w_in;
      const float *dr3 = dr2 + w_in;
      const float *dr4 = dr3 + w_in;
      const float *dr5 = dr4 + w_in;

      const float *din_ptr0 = dr0;
      const float *din_ptr1 = dr1;
      const float *din_ptr2 = dr2;
      const float *din_ptr3 = dr3;
      const float *din_ptr4 = dr4;
      const float *din_ptr5 = dr5;

      float *ptr_zero = const_cast<float *>(zero);
#ifdef __aarch64__
      for (int i = 0; i < h_out; i += 4) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;
        din_ptr5 = dr5;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        doutr2 = doutr1 + w_out;
        doutr3 = doutr2 + w_out;

        dr0 = dr4;
        dr1 = dr5;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        dr4 = dr3 + w_in;
        dr5 = dr4 + w_in;

        //! process bottom pad
        if (i + 5 >= h_in) {
          switch (i + 5 - h_in) {
            case 4:
              din_ptr1 = zero_ptr;
            case 3:
              din_ptr2 = zero_ptr;
            case 2:
              din_ptr3 = zero_ptr;
            case 1:
              din_ptr4 = zero_ptr;
            case 0:
              din_ptr5 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 4 > h_out) {
          switch (i + 4 - h_out) {
            case 3:
              doutr1 = write_ptr;
            case 2:
              doutr2 = write_ptr;
            case 1:
              doutr3 = write_ptr;
            default:
              break;
          }
        }

        int cnt = cnt_col;
        asm volatile(
            INIT_S1
            "ld1 {v8.4s}, [%[din_ptr4]], #16   \n"  /*vld1q_f32(din_ptr0)*/
            "ld1 {v10.4s}, [%[din_ptr5]], #16   \n" /*vld1q_f32(din_ptr0)*/
            "ext  v16.16b, v0.16b, v1.16b, #4 \n"   /* v16 = 1234 */
            "ext  v17.16b, v0.16b, v1.16b, #8 \n"   /* v17 = 2345 */
            "ld1 {v9.4s}, [%[din_ptr4]]   \n"       /*vld1q_f32(din_ptr0)*/
            "ld1 {v11.4s}, [%[din_ptr5]]   \n"      /*vld1q_f32(din_ptr0)*/
            MID_COMPUTE_S1 MID_RESULT_S1
            "cmp  %w[remain], #1             \n"
            "blt 0f                         \n" RIGHT_COMPUTE_S1 RIGHT_RESULT_S1
            "0: \n"
            : PARAM1
            : PARAM2, [remain] "r"(remain)
            : ASM_PARAM);
        dout_ptr = dout_ptr + 4 * w_out;
      }
#else
      for (int i = 0; i < h_out; i += 2) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = dout_ptr + w_out;

        dr0 = dr2;
        dr1 = dr3;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        //! process bottom pad
        if (i + 3 >= h_in) {
          switch (i + 3 - h_in) {
            case 2:
              din_ptr1 = zero_ptr;
            case 1:
              din_ptr2 = zero_ptr;
            case 0:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;

        unsigned int *vmask_ptr = vmask;
        asm volatile(
            INIT_S1
            "sub %[din0_ptr], #8 @ 0pad + 2 float data overlap\n"
            "sub %[din1_ptr], #8 @ 0pad + 2 float data overlap\n"
            "sub %[din2_ptr], #8 @ 0pad + 2 float data overlap\n"
            "sub %[din3_ptr], #8 @ 0pad + 2 float data overlap\n"
            "vext.32  q6, q8, q9, #1     @ 0012\n"
            "vext.32  q7, q8, q9, #2     @ 1234\n" MID_COMPUTE_S1 MID_RESULT_S1
            "cmp  %[remain], #1             \n"
            "blt 0f                         \n" RIGHT_COMPUTE_S1 RIGHT_RESULT_S1
            "0:                         \n"
            : PARAM1
            : PARAM2, [remain] "r"(remain)
            : ASM_PARAM);
        dout_ptr += 2 * w_out;
      }  //! end of processing mid rows
#endif
    }
    LITE_PARALLEL_END()
  }
}

void conv_depthwise_3x3s1p0_bias_relu(float *dout,
                                      const float *din,
                                      const float *weights,
                                      const float *bias,
                                      bool flag_bias,
                                      bool flag_relu,
                                      const int num,
                                      const int ch_in,
                                      const int h_in,
                                      const int w_in,
                                      const int h_out,
                                      const int w_out,
                                      ARMContext *ctx) {
  //! pad is done implicit
  const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  float *zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, (w_in + 6) * sizeof(float));
  float *write_ptr = zero_ptr + (w_in + 6);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  unsigned int vmask[4];
  auto &&res = right_mask_3x3s1_fp32(w_in, w_out, 0, vmask);
  uint32_t cnt_col = res.first;
  uint32_t remain = res.second;
  uint32_t right_pad_num = (4 - remain) * 4;

  float32x4_t vzero = vdupq_n_f32(0.f);

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    LITE_PARALLEL_BEGIN(c, tid, ch_in) {
      float *dout_ptr = dout_batch + c * size_out_channel;

      const float *din_ch_ptr = din_batch + c * size_in_channel;

      float bias_val = flag_bias ? bias[c] : 0.f;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};

      const float *wei_ptr = weights + c * w_stride;

      float32x4_t wr0 = vld1q_f32(wei_ptr);
      float32x4_t wr1 = vld1q_f32(wei_ptr + 3);
      float32x4_t wr2 = vld1q_f32(wei_ptr + 6);

      float *doutr0 = dout_ptr;
      float *doutr1 = doutr0 + w_out;
      float *doutr2 = doutr1 + w_out;
      float *doutr3 = doutr2 + w_out;

      const float *dr0 = din_ch_ptr;
      const float *dr1 = dr0 + w_in;
      const float *dr2 = dr1 + w_in;
      const float *dr3 = dr2 + w_in;
      const float *dr4 = dr3 + w_in;
      const float *dr5 = dr4 + w_in;

      const float *din_ptr0 = dr0;
      const float *din_ptr1 = dr1;
      const float *din_ptr2 = dr2;
      const float *din_ptr3 = dr3;
      const float *din_ptr4 = dr4;
      const float *din_ptr5 = dr5;

      float *ptr_zero = const_cast<float *>(zero);
#ifdef __aarch64__
      for (int i = 0; i < h_out; i += 4) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;
        din_ptr5 = dr5;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        doutr2 = doutr1 + w_out;
        doutr3 = doutr2 + w_out;

        dr0 = dr4;
        dr1 = dr5;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        dr4 = dr3 + w_in;
        dr5 = dr4 + w_in;

        //! process bottom pad
        if (i + 5 >= h_in) {
          switch (i + 5 - h_in) {
            case 4:
              din_ptr1 = zero_ptr;
            case 3:
              din_ptr2 = zero_ptr;
            case 2:
              din_ptr3 = zero_ptr;
            case 1:
              din_ptr4 = zero_ptr;
            case 0:
              din_ptr5 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 4 > h_out) {
          switch (i + 4 - h_out) {
            case 3:
              doutr1 = write_ptr;
            case 2:
              doutr2 = write_ptr;
            case 1:
              doutr3 = write_ptr;
            default:
              break;
          }
        }

        int cnt = cnt_col;
        asm volatile(
            INIT_S1
            "ld1 {v8.4s}, [%[din_ptr4]], #16   \n"  /*vld1q_f32(din_ptr0)*/
            "ld1 {v10.4s}, [%[din_ptr5]], #16   \n" /*vld1q_f32(din_ptr0)*/
            "ext  v16.16b, v0.16b, v1.16b, #4 \n"   /* v16 = 1234 */
            "ext  v17.16b, v0.16b, v1.16b, #8 \n"   /* v17 = 2345 */
            "ld1 {v9.4s}, [%[din_ptr4]]   \n"       /*vld1q_f32(din_ptr0)*/
            "ld1 {v11.4s}, [%[din_ptr5]]   \n"      /*vld1q_f32(din_ptr0)*/
            MID_COMPUTE_S1 MID_RESULT_S1_RELU
            "cmp  %w[remain], #1             \n"
            "blt 0f                         \n" RIGHT_COMPUTE_S1
                RIGHT_RESULT_S1_RELU "0: \n"
            : PARAM1
            : PARAM2, [remain] "r"(remain)
            : ASM_PARAM);
        dout_ptr = dout_ptr + 4 * w_out;
      }
#else
      for (int i = 0; i < h_out; i += 2) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = dout_ptr + w_out;

        dr0 = dr2;
        dr1 = dr3;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        //! process bottom pad
        if (i + 3 >= h_in) {
          switch (i + 3 - h_in) {
            case 2:
              din_ptr1 = zero_ptr;
            case 1:
              din_ptr2 = zero_ptr;
            case 0:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;

        unsigned int *vmask_ptr = vmask;
        asm volatile(INIT_S1
                     "sub %[din0_ptr], #8 @ 0pad + 2 float data overlap\n"
                     "sub %[din1_ptr], #8 @ 0pad + 2 float data overlap\n"
                     "sub %[din2_ptr], #8 @ 0pad + 2 float data overlap\n"
                     "sub %[din3_ptr], #8 @ 0pad + 2 float data overlap\n"
                     "vext.32  q6, q8, q9, #1     @ 0012\n"
                     "vext.32  q7, q8, q9, #2     @ 1234\n" MID_COMPUTE_S1
                         MID_RESULT_S1_RELU
                     "cmp  %[remain], #1             \n"
                     "blt 0f                         \n" RIGHT_COMPUTE_S1
                         RIGHT_RESULT_S1_RELU "0:                         \n"
                     : PARAM1
                     : PARAM2, [remain] "r"(remain)
                     : ASM_PARAM);
        dout_ptr += 2 * w_out;
      }  //! end of processing mid rows
#endif
    }
    LITE_PARALLEL_END()
  }
}
/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias,
 * width <= 4
 */
void conv_depthwise_3x3s1p0_bias_s_no_relu(float *dout,
                                           const float *din,
                                           const float *weights,
                                           const float *bias,
                                           bool flag_bias,
                                           bool flag_relu,
                                           const int num,
                                           const int ch_in,
                                           const int h_in,
                                           const int w_in,
                                           const int h_out,
                                           const int w_out,
                                           ARMContext *ctx) {
  //! 3x3s1 convolution, implemented by direct algorithm
  //! pad is done implicit
  //! for 4x6 convolution window
  const int right_pad_idx[8] = {5, 4, 3, 2, 1, 0, 0, 0};
  const float zero_ptr[4] = {0.f, 0.f, 0.f, 0.f};

  float32x4_t vzero = vdupq_n_f32(0.f);
  uint32x4_t vmask_rp1 =
      vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(6 - w_in));
  uint32x4_t vmask_rp2 =
      vcgeq_s32(vld1q_s32(right_pad_idx + 4), vdupq_n_s32(6 - w_in));

  unsigned int vmask[8];
  vst1q_u32(vmask, vmask_rp1);
  vst1q_u32(vmask + 4, vmask_rp2);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    LITE_PARALLEL_BEGIN(i, tid, ch_in) {
      float *dout_channel = dout_batch + i * size_out_channel;
      const float *din_channel = din_batch + i * size_in_channel;
      const float *weight_ptr = weights + i * 9;
      float32x4_t wr0 = vld1q_f32(weight_ptr);
      float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
      float32x4_t wr2 = vld1q_f32(weight_ptr + 6);

#ifdef __aarch64__
      float32x4_t wbias;
      if (flag_bias) {
        wbias = vdupq_n_f32(bias[i]);
      } else {
        wbias = vdupq_n_f32(0.f);
      }
#endif  // __aarch64__

      float out_buf1[4];
      float out_buf2[4];
      float trash_buf[4];

      float *doutr0 = dout_channel;
      float *doutr1 = dout_channel + w_out;

      for (int j = 0; j < h_out; j += 2) {
        const float *dr0 = din_channel + j * w_in;
        const float *dr1 = dr0 + w_in;
        const float *dr2 = dr1 + w_in;
        const float *dr3 = dr2 + w_in;

        doutr0 = dout_channel + j * w_out;
        doutr1 = doutr0 + w_out;

        if (j + 3 >= h_in) {
          switch (j + 3 - h_in) {
            case 2:
              dr1 = zero_ptr;
            case 1:
              dr2 = zero_ptr;
              doutr1 = trash_buf;
            case 0:
              dr3 = zero_ptr;
              if (j + 2 > h_out) {
                doutr1 = trash_buf;
              }
            default:
              break;
          }
        }
#ifdef __aarch64__
        asm volatile(COMPUTE_S_S1_P0 RESULT_S_S1
                     : [din0] "+r"(dr0),
                       [din1] "+r"(dr1),
                       [din2] "+r"(dr2),
                       [din3] "+r"(dr3)
                     : [wr0] "w"(wr0),
                       [wr1] "w"(wr1),
                       [wr2] "w"(wr2),
                       [vbias] "w"(wbias),
                       [mask1] "w"(vmask_rp1),
                       [mask2] "w"(vmask_rp2),
                       [zero] "w"(vzero),
                       [out1] "r"(out_buf1),
                       [out2] "r"(out_buf2)
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
                       "v15");
#else
        unsigned int *vmask_ptr = vmask;
        float bias_val = flag_bias ? bias[i] : 0.f;
        asm volatile(COMPUTE_S_S1_P0 RESULT_S_S1
                     : [din0] "+r"(dr0),
                       [din1] "+r"(dr1),
                       [din2] "+r"(dr2),
                       [din3] "+r"(dr3),
                       [vmask] "+r"(vmask_ptr)
                     : [wr0] "w"(wr0),
                       [wr1] "w"(wr1),
                       [wr2] "w"(wr2),
                       [vzero] "w"(vzero),
                       [bias_val] "r"(bias_val),
                       [out1] "r"(out_buf1),
                       [out2] "r"(out_buf2)
                     : "cc",
                       "memory",
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
        for (int w = 0; w < w_out; ++w) {
          *doutr0++ = out_buf1[w];
          *doutr1++ = out_buf2[w];
        }
      }  // end of processing heights
    }    // end of processing channels
    LITE_PARALLEL_END()
  }  // end of processing batchs
}

void conv_depthwise_3x3s1p0_bias_s_relu(float *dout,
                                        const float *din,
                                        const float *weights,
                                        const float *bias,
                                        bool flag_bias,
                                        bool flag_relu,
                                        const int num,
                                        const int ch_in,
                                        const int h_in,
                                        const int w_in,
                                        const int h_out,
                                        const int w_out,
                                        ARMContext *ctx) {
  //! 3x3s1 convolution, implemented by direct algorithm
  //! pad is done implicit
  //! for 4x6 convolution window
  const int right_pad_idx[8] = {5, 4, 3, 2, 1, 0, 0, 0};
  const float zero_ptr[4] = {0.f, 0.f, 0.f, 0.f};

  float32x4_t vzero = vdupq_n_f32(0.f);
  uint32x4_t vmask_rp1 =
      vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(6 - w_in));
  uint32x4_t vmask_rp2 =
      vcgeq_s32(vld1q_s32(right_pad_idx + 4), vdupq_n_s32(6 - w_in));

  unsigned int vmask[8];
  vst1q_u32(vmask, vmask_rp1);
  vst1q_u32(vmask + 4, vmask_rp2);

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * ch_in * size_in_channel;
    float *dout_batch = dout + n * ch_in * size_out_channel;

    LITE_PARALLEL_BEGIN(i, tid, ch_in) {
      float *dout_channel = dout_batch + i * size_out_channel;
      const float *din_channel = din_batch + i * size_in_channel;
      const float *weight_ptr = weights + i * 9;
      float32x4_t wr0 = vld1q_f32(weight_ptr);
      float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
      float32x4_t wr2 = vld1q_f32(weight_ptr + 6);

#ifdef __aarch64__
      float32x4_t wbias;
      if (flag_bias) {
        wbias = vdupq_n_f32(bias[i]);
      } else {
        wbias = vdupq_n_f32(0.f);
      }
#endif  // __aarch64__

      float out_buf1[4];
      float out_buf2[4];
      float trash_buf[4];

      float *doutr0 = dout_channel;
      float *doutr1 = dout_channel + w_out;

      for (int j = 0; j < h_out; j += 2) {
        const float *dr0 = din_channel + j * w_in;
        const float *dr1 = dr0 + w_in;
        const float *dr2 = dr1 + w_in;
        const float *dr3 = dr2 + w_in;

        doutr0 = dout_channel + j * w_out;
        doutr1 = doutr0 + w_out;

        if (j + 3 >= h_in) {
          switch (j + 3 - h_in) {
            case 2:
              dr1 = zero_ptr;
            case 1:
              dr2 = zero_ptr;
              doutr1 = trash_buf;
            case 0:
              dr3 = zero_ptr;
              if (j + 2 > h_out) {
                doutr1 = trash_buf;
              }
            default:
              break;
          }
        }
#ifdef __aarch64__
        asm volatile(COMPUTE_S_S1_P0 RESULT_S_S1_RELU
                     : [din0] "+r"(dr0),
                       [din1] "+r"(dr1),
                       [din2] "+r"(dr2),
                       [din3] "+r"(dr3)
                     : [wr0] "w"(wr0),
                       [wr1] "w"(wr1),
                       [wr2] "w"(wr2),
                       [vbias] "w"(wbias),
                       [mask1] "w"(vmask_rp1),
                       [mask2] "w"(vmask_rp2),
                       [zero] "w"(vzero),
                       [out1] "r"(out_buf1),
                       [out2] "r"(out_buf2)
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
                       "v15");
#else
        unsigned int *vmask_ptr = vmask;
        float bias_val = flag_bias ? bias[i] : 0.f;
        asm volatile(COMPUTE_S_S1_P0 RESULT_S_S1_RELU
                     : [din0] "+r"(dr0),
                       [din1] "+r"(dr1),
                       [din2] "+r"(dr2),
                       [din3] "+r"(dr3),
                       [vmask] "+r"(vmask_ptr)
                     : [wr0] "w"(wr0),
                       [wr1] "w"(wr1),
                       [wr2] "w"(wr2),
                       [vzero] "w"(vzero),
                       [bias_val] "r"(bias_val),
                       [out1] "r"(out_buf1),
                       [out2] "r"(out_buf2)
                     : "cc",
                       "memory",
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
        for (int w = 0; w < w_out; ++w) {
          *doutr0++ = out_buf1[w];
          *doutr1++ = out_buf2[w];
        }
      }  // end of processing heights
    }    // end of processing channels
    LITE_PARALLEL_END()
  }  // end of processing batchs
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
