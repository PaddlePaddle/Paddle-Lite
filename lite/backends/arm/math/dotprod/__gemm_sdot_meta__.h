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
#define GEMM_SDOT_INT8_KERNEL                                              \
  "ldp    q0, q1, [%[a_ptr]], #32\n"     /* load a00,a01 to q0, q1*/       \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b0, b1 to q4, q5*/        \
  "eor    v8.16b,  v8.16b, v8.16b\n"     /* out0 = 0 */                    \
  "eor    v9.16b,  v9.16b, v9.16b\n"     /* out1 = 0 */                    \
  "eor    v10.16b,  v10.16b, v10.16b\n"  /* out2 = 0 */                    \
  "eor    v11.16b,  v11.16b, v11.16b\n"  /* out3 = 0 */                    \
  "eor    v12.16b,  v12.16b, v12.16b\n"  /* out4 = 0 */                    \
  "prfm   pldl1keep, [%[b_ptr], #64]\n"  /* preload b*/                    \
  "eor    v13.16b,  v13.16b, v13.16b\n"  /* out5 = 0 */                    \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"  /* preload a*/                    \
  "eor    v14.16b,  v14.16b, v14.16b\n"  /* out6 = 0 */                    \
  "prfm   pldl1keep, [%[b_ptr], #128]\n" /* preload b*/                    \
  "eor    v15.16b,  v15.16b, v15.16b\n"  /* out7 = 0 */                    \
  "prfm   pldl1keep, [%[a_ptr], #128]\n" /* preload a*/                    \
  "eor    v16.16b,  v16.16b, v16.16b\n"  /* out8 = 0 */                    \
  "prfm   pldl1keep, [%[b_ptr], #192]\n" /* preload b*/                    \
  "eor    v17.16b,  v17.16b, v17.16b\n"  /* out9 = 0 */                    \
  "prfm   pldl1keep, [%[b_ptr], #256]\n" /* preload b*/                    \
  "eor    v18.16b,  v18.16b, v18.16b\n"  /* out10 = 0 */                   \
  "prfm   pldl1keep, [%[a_ptr], #192]\n" /* preload a*/                    \
  "eor    v19.16b,  v19.16b, v19.16b\n"  /* out11 = 0 */                   \
  "prfm   pldl1keep, [%[b_ptr], #320]\n" /* preload b*/                    \
  "eor    v20.16b,  v20.16b, v20.16b\n"  /* out12 = 0 */                   \
  "prfm   pldl1keep, [%[a_ptr], #256]\n" /* preload a*/                    \
  "eor    v21.16b,  v21.16b, v21.16b\n"  /* out13 = 0 */                   \
  "prfm   pldl1keep, [%[b_ptr], #384]\n" /* preload b*/                    \
  "eor    v22.16b,  v22.16b, v22.16b\n"  /* out14 = 0 */                   \
  "eor    v23.16b,  v23.16b, v23.16b\n"  /* out15 = 0 */                   \
  "eor    v24.16b,  v24.16b, v24.16b\n"  /* out16 = 0 */                   \
  "eor    v25.16b,  v25.16b, v25.16b\n"  /* out17 = 0 */                   \
  "eor    v26.16b,  v26.16b, v26.16b\n"  /* out18 = 0 */                   \
  "eor    v27.16b,  v27.16b, v27.16b\n"  /* out19 = 0 */                   \
  "eor    v28.16b,  v28.16b, v28.16b\n"  /* out20 = 0 */                   \
  "eor    v29.16b,  v29.16b, v29.16b\n"  /* out21 = 0 */                   \
  "eor    v30.16b,  v30.16b, v30.16b\n"  /* out22 = 0 */                   \
  "eor    v31.16b,  v31.16b, v31.16b\n"  /* out23 = 0 */                   \
  "cbz    %w[k], 2f\n" /* check loop count > 0 */                          \
  /* main loop, unrool 0*/                                                 \
  "1:\n"                                 /* main loop */                   \
  "sdot   v8.4s ,  v4.16b,  v0.4b[0]\n"  /* out0 = b0 * a00[0], b0 = q4 */ \
  "sdot   v11.4s ,  v4.16b,  v0.4b[1]\n" /* out1 = b0 * a00[1], b0 = q4 */ \
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b2, b0 to q6, q7       */ \
  "sdot   v14.4s,  v4.16b,  v0.4b[2]\n"  /* out2 = b0 * a00[2], b0 = q4 */ \
  "sdot   v17.4s,  v4.16b,  v0.4b[3]\n"  /* out3 = b0 * a00[3], b0 = q4 */ \
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q3, q4     */ \
  "sdot   v20.4s,  v4.16b,  v1.4b[0]\n"  /* out4 = b0 * a01[0], b0 = q4 */ \
  "sdot   v23.4s,  v4.16b,  v1.4b[1]\n"  /* out5 = b0 * a01[1], b0 = q4 */ \
  "sdot   v26.4s,  v4.16b,  v1.4b[2]\n"  /* out6 = b0 * a01[2], b0 = q4 */ \
  "sdot   v29.4s,  v4.16b,  v1.4b[3]\n"  /* out7 = b0 * a01[3], b0 = q4 */ \
  "sdot   v9.4s,  v5.16b,  v0.4b[0]\n"   /* out8 = b1 * a00[0], b1 = q5 */ \
  "sdot   v12.4s,  v5.16b,  v0.4b[1]\n"  /* out9 = b1 * a00[1], b1 = q5 */ \
  "sdot   v15.4s,  v5.16b,  v0.4b[2]\n"  /* out10 = b1 * a00[2], b1 = q5*/ \
  "sdot   v18.4s,  v5.16b,  v0.4b[3]\n"  /* out11 = b1 * a00[3], b1 = q5*/ \
  "sdot   v21.4s,  v5.16b,  v1.4b[0]\n"  /* out12 = b1 * a01[0], b1 = q5*/ \
  "sdot   v24.4s,  v5.16b,  v1.4b[1]\n"  /* out13 = b1 * a01[1], b1 = q5*/ \
  "sdot   v27.4s,  v5.16b,  v1.4b[2]\n"  /* out14 = b1 * a01[2], b1 = q5*/ \
  "sdot   v30.4s,  v5.16b,  v1.4b[3]\n"  /* out15 = b1 * a01[3], b1 = q5*/ \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b1, b2 to q4, q5       */ \
  "sdot   v10.4s,  v6.16b,  v0.4b[0]\n"  /* out16 = b2 * a00[0], b2 = q6*/ \
  "sdot   v13.4s,  v6.16b,  v0.4b[1]\n"  /* out17 = b2 * a00[1], b2 = q6*/ \
  "prfm   pldl1keep, [%[b_ptr], #384]\n"                                   \
  "sdot   v16.4s,  v6.16b,  v0.4b[2]\n" /* out18 = b2 * a00[2], b2 = q6*/  \
  "sdot   v19.4s,  v6.16b,  v0.4b[3]\n" /* out19 = b2 * a00[3], b2 = q6*/  \
  "sdot   v22.4s,  v6.16b,  v1.4b[0]\n" /* out20 = b2 * a00[0], b2 = q6*/  \
  "sdot   v25.4s,  v6.16b,  v1.4b[1]\n" /* out21 = b2 * a00[1], b2 = q6*/  \
  "sdot   v28.4s,  v6.16b,  v1.4b[2]\n" /* out22 = b2 * a00[2], b2 = q6*/  \
  "sdot   v31.4s,  v6.16b,  v1.4b[3]\n" /* out23 = b2 * a00[3], b2 = q6*/  \
  "ldp    q0, q1, [%[a_ptr]], #32\n"    /* load a00, a01 to q0, q1 */      \
  /* unrool 1 */                                                           \
  "sdot   v8.4s ,  v7.16b,  v2.4b[0]\n" /* out0 = b0 * a10[0], b0 = q7 */  \
  "sdot   v11.4s ,  v7.16b,  v2.4b[1]\n"/* out1 = b0 * a10[1], b0 = q7 */  \
  "sdot   v14.4s,  v7.16b,  v2.4b[2]\n" /* out2 = b0 * a10[2], b0 = q7 */  \
  "prfm   pldl1keep, [%[a_ptr], #256]\n"                                   \
  "sdot   v17.4s,  v7.16b,  v2.4b[3]\n" /* out3 = b0 * a10[3], b0 = q7 */  \
  "sdot   v20.4s,  v7.16b,  v3.4b[0]\n" /* out4 = b0 * a11[0], b0 = q7 */  \
  "sdot   v23.4s,  v7.16b,  v3.4b[1]\n" /* out5 = b0 * a11[1], b0 = q7 */  \
  "sdot   v26.4s,  v7.16b,  v3.4b[2]\n" /* out6 = b0 * a11[2], b0 = q7 */  \
  "sdot   v29.4s,  v7.16b,  v3.4b[3]\n" /* out7 = b0 * a11[3], b0 = q7 */  \
  "ldp    q6, q7, [%[b_ptr]], #32\n"    /* load b0, b1 to q6, q7       */  \
  "sdot   v9.4s,  v4.16b,  v2.4b[0]\n"  /* out8 = b0 * a10[0], b1 = q4 */  \
  "sdot   v12.4s,  v4.16b,  v2.4b[1]\n" /* out9 = b0 * a10[1], b1 = q4 */  \
  "sdot   v15.4s,  v4.16b,  v2.4b[2]\n" /* out10 = b1 * a10[2], b1 = q4*/  \
  "sdot   v18.4s,  v4.16b,  v2.4b[3]\n" /* out11 = b1 * a10[3], b1 = q4*/  \
  "sdot   v21.4s,  v4.16b,  v3.4b[0]\n" /* out12 = b1 * a10[0], b1 = q4*/  \
  "sdot   v24.4s,  v4.16b,  v3.4b[1]\n" /* out13 = b1 * a10[1], b1 = q4*/  \
  "sdot   v27.4s,  v4.16b,  v3.4b[2]\n" /* out14 = b1 * a10[2], b1 = q4*/  \
  "sdot   v30.4s,  v4.16b,  v3.4b[3]\n" /* out15 = b1 * a10[3], b1 = q4*/  \
  "sdot   v10.4s,  v5.16b,  v2.4b[0]\n" /* out16 = b2 * a10[0], b2 = q5*/  \
  "sdot   v13.4s,  v5.16b,  v2.4b[1]\n" /* out17 = b2 * a10[0], b2 = q5*/  \
  "sdot   v16.4s,  v5.16b,  v2.4b[2]\n" /* out18 = b2 * a10[0], b2 = q5*/  \
  "sdot   v19.4s,  v5.16b,  v2.4b[3]\n" /* out19 = b2 * a10[0], b2 = q5*/  \
  "sdot   v22.4s,  v5.16b,  v3.4b[0]\n" /* out20 = b2 * a10[0], b2 = q5*/  \
  "sdot   v25.4s,  v5.16b,  v3.4b[1]\n" /* out21 = b2 * a10[0], b2 = q5*/  \
  "sdot   v28.4s,  v5.16b,  v3.4b[2]\n" /* out22 = b2 * a10[0], b2 = q5*/  \
  "sdot   v31.4s,  v5.16b,  v3.4b[3]\n" /* out23 = b2 * a10[0], b2 = q5*/  \
  "ldp    q4, q5, [%[b_ptr]], #32\n"    /* load b2, b0 to q4, q5 */        \
  /* unrool 2*/                                                            \
  "sdot   v8.4s ,  v6.16b,  v0.4b[0]\n"  /* out0 = b0 * a00[0], b0 = q6 */ \
  "sdot   v11.4s ,  v6.16b,  v0.4b[1]\n" /* out1 = b0 * a00[1], b0 = q6 */ \
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q3, q4*/      \
  "sdot   v14.4s,  v6.16b,  v0.4b[2]\n"  /* out2 = b0 * a00[2], b0 = q6*/  \
  "sdot   v17.4s,  v6.16b,  v0.4b[3]\n"  /* out3 = b0 * a00[3], b0 = q6*/  \
  "sdot   v20.4s,  v6.16b,  v1.4b[0]\n"  /* out4 = b0 * a01[0], b0 = q6*/  \
  "sdot   v23.4s,  v6.16b,  v1.4b[1]\n"  /* out5 = b0 * a01[1], b0 = q6*/  \
  "sdot   v26.4s,  v6.16b,  v1.4b[2]\n"  /* out6 = b0 * a01[2], b0 = q6*/  \
  "sdot   v29.4s,  v6.16b,  v1.4b[3]\n"  /* out7 = b0 * a01[3], b0 = q6*/  \
  "sdot   v9.4s,  v7.16b,  v0.4b[0]\n"   /* out8 = b1 * a00[0], b1 = q7*/  \
  "sdot   v12.4s,  v7.16b,  v0.4b[1]\n"  /* out9 = b1 * a00[1], b1 = q7*/  \
  "prfm   pldl1keep, [%[b_ptr], #384]\n"                                   \
  "sdot   v15.4s,  v7.16b,  v0.4b[2]\n" /* out10 = b1 * a00[2], b1 = q7*/  \
  "sdot   v18.4s,  v7.16b,  v0.4b[3]\n" /* out11 = b1 * a00[3], b1 = q7*/  \
  "sdot   v21.4s,  v7.16b,  v1.4b[0]\n" /* out12 = b1 * a01[0], b1 = q7*/  \
  "sdot   v24.4s,  v7.16b,  v1.4b[1]\n" /* out13 = b1 * a01[1], b1 = q7*/  \
  "sdot   v27.4s,  v7.16b,  v1.4b[2]\n" /* out14 = b1 * a01[2], b1 = q7*/  \
  "sdot   v30.4s,  v7.16b,  v1.4b[3]\n" /* out15 = b1 * a01[3], b1 = q7*/  \
  "ldp    q6, q7, [%[b_ptr]], #32\n"    /* load b1, b2 to q6, q7*/         \
  "sdot   v10.4s,  v4.16b,  v0.4b[0]\n" /* out16 = b2 * a00[0], b2 = q4*/  \
  "sdot   v13.4s,  v4.16b,  v0.4b[1]\n" /* out17 = b2 * a00[1], b2 = q4*/  \
  "sdot   v16.4s,  v4.16b,  v0.4b[2]\n" /* out18 = b2 * a00[2], b2 = q4*/  \
  "sdot   v19.4s,  v4.16b,  v0.4b[3]\n" /* out19 = b2 * a00[3], b2 = q4*/  \
  "sdot   v22.4s,  v4.16b,  v1.4b[0]\n" /* out20 = b2 * a00[0], b2 = q4*/  \
  "sdot   v25.4s,  v4.16b,  v1.4b[1]\n" /* out21 = b2 * a00[1], b2 = q4*/  \
  "sdot   v28.4s,  v4.16b,  v1.4b[2]\n" /* out22 = b2 * a00[2], b2 = q4*/  \
  "sdot   v31.4s,  v4.16b,  v1.4b[3]\n" /* out23 = b2 * a00[3], b2 = q4*/  \
  "ldp    q0, q1, [%[a_ptr]], #32\n" /* load a00, a01 to q0, q1*/          \
  /* unrool 3*/                                                            \
  "sdot   v8.4s ,  v5.16b,  v2.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q5*/  \
  "sdot   v11.4s ,  v5.16b,  v2.4b[1]\n" /* out1 = b0 * a10[1], b0 = q5*/  \
  "sdot   v14.4s,  v5.16b,  v2.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q5*/  \
  "sdot   v17.4s,  v5.16b,  v2.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q5*/  \
  "sdot   v20.4s,  v5.16b,  v3.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q5*/  \
  "sdot   v23.4s,  v5.16b,  v3.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q5*/  \
  "sdot   v26.4s,  v5.16b,  v3.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q5*/  \
  "sdot   v29.4s,  v5.16b,  v3.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q5*/  \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b0, b1 to q4, q5*/        \
  "sdot   v9.4s,  v6.16b,  v2.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q6*/  \
  "sdot   v12.4s,  v6.16b,  v2.4b[1]\n"  /* out9 = b0 * a10[1], b1 = q6*/  \
  "prfm   pldl1keep, [%[a_ptr], #256]\n"                                   \
  "sdot   v15.4s,  v6.16b,  v2.4b[2]\n" /* out10 = b1 * a10[2], b1 = q6*/  \
  "sdot   v18.4s,  v6.16b,  v2.4b[3]\n" /* out11 = b1 * a10[3], b1 = q6*/  \
  "sdot   v21.4s,  v6.16b,  v3.4b[0]\n" /* out12 = b1 * a10[0], b1 = q6*/  \
  "sdot   v24.4s,  v6.16b,  v3.4b[1]\n" /* out13 = b1 * a10[1], b1 = q6*/  \
  "sdot   v27.4s,  v6.16b,  v3.4b[2]\n" /* out14 = b1 * a10[2], b1 = q6*/  \
  "prfm   pldl1keep, [%[b_ptr], #384]\n"                                   \
  "sdot   v30.4s,  v6.16b,  v3.4b[3]\n" /* out15 = b1 * a10[3], b1 = q6*/  \
  "sdot   v10.4s,  v7.16b,  v2.4b[0]\n" /* out16 = b2 * a10[0], b2 = q7*/  \
  "sdot   v13.4s,  v7.16b,  v2.4b[1]\n" /* out17 = b2 * a10[0], b2 = q7*/  \
  "sdot   v16.4s,  v7.16b,  v2.4b[2]\n" /* out18 = b2 * a10[0], b2 = q7*/  \
  "sdot   v19.4s,  v7.16b,  v2.4b[3]\n" /* out19 = b2 * a10[0], b2 = q7*/  \
  "sdot   v22.4s,  v7.16b,  v3.4b[0]\n" /* out20 = b2 * a10[0], b2 = q7*/  \
  "sdot   v25.4s,  v7.16b,  v3.4b[1]\n" /* out21 = b2 * a10[0], b2 = q7*/  \
  "subs   %w[k], %w[k], #1\n"           /* loop count - 1*/                \
  "sdot   v28.4s,  v7.16b,  v3.4b[2]\n" /* out22 = b2 * a10[0], b2 = q7*/  \
  "sdot   v31.4s,  v7.16b,  v3.4b[3]\n" /* out23 = b2 * a10[0], b2 = q7*/  \
  "bne    1b\n" /* Target to use when K is 1 or 2 */                       \
  "2:\n"                                             /* process tail*/     \
  "subs       %w[tail], %w[tail], #1\n"              /* tail--*/           \
  "beq        3f\n" /*jump to tail = 1*/                                   \
  /* final unrool 0, unrool 0, tail > 1*/                                  \
  "sdot   v8.4s ,  v4.16b,  v0.4b[0]\n"  /* out0 = b0 * a00[0], b0 = q4*/  \
  "sdot   v11.4s ,  v4.16b,  v0.4b[1]\n" /* out1 = b0 * a00[1], b0 = q4*/  \
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b2, b0 to q6, q7*/        \
  "sdot   v14.4s,  v4.16b,  v0.4b[2]\n"  /* out2 = b0 * a00[2], b0 = q4*/  \
  "sdot   v17.4s,  v4.16b,  v0.4b[3]\n"  /* out3 = b0 * a00[3], b0 = q4*/  \
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q2, q3*/      \
  "sdot   v20.4s,  v4.16b,  v1.4b[0]\n"  /* out4 = b0 * a01[0], b0 = q4*/  \
  "sdot   v23.4s,  v4.16b,  v1.4b[1]\n"  /* out5 = b0 * a01[1], b0 = q4*/  \
  "sdot   v26.4s,  v4.16b,  v1.4b[2]\n"  /* out6 = b0 * a01[2], b0 = q4*/  \
  "sdot   v29.4s,  v4.16b,  v1.4b[3]\n"  /* out7 = b0 * a01[3], b0 = q4*/  \
  "subs   %w[tail], %w[tail], #1\n"      /* tail--*/                       \
  "sdot   v9.4s,  v5.16b,  v0.4b[0]\n"   /* out8 = b1 * a00[0], b1 = q5*/  \
  "sdot   v12.4s,  v5.16b,  v0.4b[1]\n"  /* out9 = b1 * a00[1], b1 = q5*/  \
  "sdot   v15.4s,  v5.16b,  v0.4b[2]\n"  /* out10 = b1 * a00[2], b1 = q5*/ \
  "sdot   v18.4s,  v5.16b,  v0.4b[3]\n"  /* out11 = b1 * a00[3], b1 = q5*/ \
  "sdot   v21.4s,  v5.16b,  v1.4b[0]\n"  /* out12 = b1 * a01[0], b1 = q5*/ \
  "sdot   v24.4s,  v5.16b,  v1.4b[1]\n"  /* out13 = b1 * a01[1], b1 = q5*/ \
  "sdot   v27.4s,  v5.16b,  v1.4b[2]\n"  /* out14 = b1 * a01[2], b1 = q5*/ \
  "sdot   v30.4s,  v5.16b,  v1.4b[3]\n"  /* out15 = b1 * a01[3], b1 = q5*/ \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b1, b2 to q4, q5*/        \
  "sdot   v10.4s,  v6.16b,  v0.4b[0]\n"  /* out16 = b2 * a00[0], b2 = q6*/ \
  "sdot   v13.4s,  v6.16b,  v0.4b[1]\n"  /* out17 = b2 * a00[1], b2 = q6*/ \
  "sdot   v16.4s,  v6.16b,  v0.4b[2]\n"  /* out18 = b2 * a00[2], b2 = q6*/ \
  "sdot   v19.4s,  v6.16b,  v0.4b[3]\n"  /* out19 = b2 * a00[3], b2 = q6*/ \
  "sdot   v22.4s,  v6.16b,  v1.4b[0]\n"  /* out20 = b2 * a00[0], b2 = q6*/ \
  "sdot   v25.4s,  v6.16b,  v1.4b[1]\n"  /* out21 = b2 * a00[1], b2 = q6*/ \
  "sdot   v28.4s,  v6.16b,  v1.4b[2]\n"  /* out22 = b2 * a00[2], b2 = q6*/ \
  "sdot   v31.4s,  v6.16b,  v1.4b[3]\n"  /* out23 = b2 * a00[3], b2 = q6*/ \
  "beq        4f\n" /*jump to tail = 2*/                                   \
  /* unrool 1, tail > 2*/                                                  \
  "ldp    q0, q1, [%[a_ptr]], #32\n"     /* load a00, a01 to q0, q1*/      \
  "sdot   v8.4s ,  v7.16b,  v2.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q7*/  \
  "sdot   v11.4s ,  v7.16b,  v2.4b[1]\n" /* out1 = b0 * a10[1], b0 = q7*/  \
  "sdot   v14.4s,  v7.16b,  v2.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q7*/  \
  "sdot   v17.4s,  v7.16b,  v2.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q7*/  \
  "sdot   v20.4s,  v7.16b,  v3.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q7*/  \
  "sdot   v23.4s,  v7.16b,  v3.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q7*/  \
  "sdot   v26.4s,  v7.16b,  v3.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q7*/  \
  "sdot   v29.4s,  v7.16b,  v3.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q7*/  \
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b0, b1 to q6, q7*/        \
  "sdot   v9.4s,  v4.16b,  v2.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q4*/  \
  "sdot   v12.4s,  v4.16b,  v2.4b[1]\n"  /* out9 = b0 * a10[1], b1 = q4*/  \
  "sdot   v15.4s,  v4.16b,  v2.4b[2]\n"  /* out10 = b1 * a10[2], b1 = q4*/ \
  "sdot   v18.4s,  v4.16b,  v2.4b[3]\n"  /* out11 = b1 * a10[3], b1 = q4*/ \
  "sdot   v21.4s,  v4.16b,  v3.4b[0]\n"  /* out12 = b1 * a10[0], b1 = q4*/ \
  "sdot   v24.4s,  v4.16b,  v3.4b[1]\n"  /* out13 = b1 * a10[1], b1 = q4*/ \
  "sdot   v27.4s,  v4.16b,  v3.4b[2]\n"  /* out14 = b1 * a10[2], b1 = q4*/ \
  "sdot   v30.4s,  v4.16b,  v3.4b[3]\n"  /* out15 = b1 * a10[3], b1 = q4*/ \
  "subs   %w[tail], %w[tail], #1\n"      /* tail--*/                       \
  "sdot   v10.4s,  v5.16b,  v2.4b[0]\n"  /* out16 = b2 * a10[0], b2 = q5*/ \
  "sdot   v13.4s,  v5.16b,  v2.4b[1]\n"  /* out17 = b2 * a10[0], b2 = q5*/ \
  "sdot   v16.4s,  v5.16b,  v2.4b[2]\n"  /* out18 = b2 * a10[0], b2 = q5*/ \
  "sdot   v19.4s,  v5.16b,  v2.4b[3]\n"  /* out19 = b2 * a10[0], b2 = q5*/ \
  "sdot   v22.4s,  v5.16b,  v3.4b[0]\n"  /* out20 = b2 * a10[0], b2 = q5*/ \
  "sdot   v25.4s,  v5.16b,  v3.4b[1]\n"  /* out21 = b2 * a10[0], b2 = q5*/ \
  "sdot   v28.4s,  v5.16b,  v3.4b[2]\n"  /* out22 = b2 * a10[0], b2 = q5*/ \
  "sdot   v31.4s,  v5.16b,  v3.4b[3]\n"  /* out23 = b2 * a10[0], b2 = q5*/ \
  "beq        5f\n" /*jump to tail = 3*/                                   \
  /* unrool 2, tail = 4*/                                                  \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b2, b0 to q4, q5*/        \
  "sdot   v8.4s ,  v6.16b,  v0.4b[0]\n"  /* out0 = b0 * a00[0], b0 = q6*/  \
  "sdot   v11.4s ,  v6.16b,  v0.4b[1]\n" /* out1 = b0 * a00[1], b0 = q6*/  \
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q3, q4*/      \
  "sdot   v14.4s,  v6.16b,  v0.4b[2]\n"  /* out2 = b0 * a00[2], b0 = q6*/  \
  "sdot   v17.4s,  v6.16b,  v0.4b[3]\n"  /* out3 = b0 * a00[3], b0 = q6*/  \
  "sdot   v20.4s,  v6.16b,  v1.4b[0]\n"  /* out4 = b0 * a01[0], b0 = q6*/  \
  "sdot   v23.4s,  v6.16b,  v1.4b[1]\n"  /* out5 = b0 * a01[1], b0 = q6*/  \
  "sdot   v26.4s,  v6.16b,  v1.4b[2]\n"  /* out6 = b0 * a01[2], b0 = q6*/  \
  "sdot   v29.4s,  v6.16b,  v1.4b[3]\n"  /* out7 = b0 * a01[3], b0 = q6*/  \
  "sdot   v9.4s,  v7.16b,  v0.4b[0]\n"   /* out8 = b1 * a00[0], b1 = q7*/  \
  "sdot   v12.4s,  v7.16b,  v0.4b[1]\n"  /* out9 = b1 * a00[1], b1 = q7*/  \
  "sdot   v15.4s,  v7.16b,  v0.4b[2]\n"  /* out10 = b1 * a00[2], b1 = q7*/ \
  "sdot   v18.4s,  v7.16b,  v0.4b[3]\n"  /* out11 = b1 * a00[3], b1 = q7*/ \
  "sdot   v21.4s,  v7.16b,  v1.4b[0]\n"  /* out12 = b1 * a01[0], b1 = q7*/ \
  "sdot   v24.4s,  v7.16b,  v1.4b[1]\n"  /* out13 = b1 * a01[1], b1 = q7*/ \
  "sdot   v27.4s,  v7.16b,  v1.4b[2]\n"  /* out14 = b1 * a01[2], b1 = q7*/ \
  "sdot   v30.4s,  v7.16b,  v1.4b[3]\n"  /* out15 = b1 * a01[3], b1 = q7*/ \
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b1, b2 to q6, q7*/        \
  "sdot   v10.4s,  v4.16b,  v0.4b[0]\n"  /* out16 = b2 * a00[0], b2 = q4*/ \
  "sdot   v13.4s,  v4.16b,  v0.4b[1]\n"  /* out17 = b2 * a00[1], b2 = q4*/ \
  "sdot   v16.4s,  v4.16b,  v0.4b[2]\n"  /* out18 = b2 * a00[2], b2 = q4*/ \
  "sdot   v19.4s,  v4.16b,  v0.4b[3]\n"  /* out19 = b2 * a00[3], b2 = q4*/ \
  "sdot   v22.4s,  v4.16b,  v1.4b[0]\n"  /* out20 = b2 * a00[0], b2 = q4*/ \
  "sdot   v25.4s,  v4.16b,  v1.4b[1]\n"  /* out21 = b2 * a00[1], b2 = q4*/ \
  "sdot   v28.4s,  v4.16b,  v1.4b[2]\n"  /* out22 = b2 * a00[2], b2 = q4*/ \
  "sdot   v31.4s,  v4.16b,  v1.4b[3]\n"  /* out23 = b2 * a00[3], b2 = q4*/ \
  /* unrool 3, tail = 4*/                                                  \
  "sdot   v8.4s ,  v5.16b,  v2.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q5*/  \
  "sdot   v11.4s ,  v5.16b,  v2.4b[1]\n" /* out1 = b0 * a10[1], b0 = q5*/  \
  "sdot   v14.4s,  v5.16b,  v2.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q5*/  \
  "sdot   v17.4s,  v5.16b,  v2.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q5*/  \
  "sdot   v20.4s,  v5.16b,  v3.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q5*/  \
  "sdot   v23.4s,  v5.16b,  v3.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q5*/  \
  "sdot   v26.4s,  v5.16b,  v3.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q5*/  \
  "sdot   v29.4s,  v5.16b,  v3.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q5*/  \
  "sdot   v9.4s,  v6.16b,  v2.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q6*/  \
  "sdot   v12.4s,  v6.16b,  v2.4b[1]\n"  /* out9 = b1 * a10[1], b1 = q6*/  \
  "sdot   v15.4s,  v6.16b,  v2.4b[2]\n"  /* out10 = b1 * a10[2], b1 = q6*/ \
  "sdot   v18.4s,  v6.16b,  v2.4b[3]\n"  /* out11 = b1 * a10[3], b1 = q6*/ \
  "sdot   v21.4s,  v6.16b,  v3.4b[0]\n"  /* out12 = b1 * a10[0], b1 = q6*/ \
  "sdot   v24.4s,  v6.16b,  v3.4b[1]\n"  /* out13 = b1 * a10[1], b1 = q6*/ \
  "sdot   v27.4s,  v6.16b,  v3.4b[2]\n"  /* out14 = b1 * a10[2], b1 = q6*/ \
  "sdot   v30.4s,  v6.16b,  v3.4b[3]\n"  /* out15 = b1 * a10[3], b1 = q6*/ \
  "sdot   v10.4s,  v7.16b,  v2.4b[0]\n"  /* out16 = b2 * a10[0], b2 = q7*/ \
  "sdot   v13.4s,  v7.16b,  v2.4b[1]\n"  /* out17 = b2 * a10[0], b2 = q7*/ \
  "sdot   v16.4s,  v7.16b,  v2.4b[2]\n"  /* out18 = b2 * a10[0], b2 = q7*/ \
  "sdot   v19.4s,  v7.16b,  v2.4b[3]\n"  /* out19 = b2 * a10[0], b2 = q7*/ \
  "sdot   v22.4s,  v7.16b,  v3.4b[0]\n"  /* out20 = b2 * a10[0], b2 = q7*/ \
  "sdot   v25.4s,  v7.16b,  v3.4b[1]\n"  /* out21 = b2 * a10[0], b2 = q7*/ \
  "sdot   v28.4s,  v7.16b,  v3.4b[2]\n"  /* out22 = b2 * a10[0], b2 = q7*/ \
  "sdot   v31.4s,  v7.16b,  v3.4b[3]\n"  /* out23 = b2 * a10[0], b2 = q7*/ \
  "b      11f\n"                         /* tails==1 final tail*/          \
  "3: \n"                                /* tail=1*/                       \
  "ldr    q6, [%[b_ptr]], #16\n"         /* load b2 to q6*/                \
  "sdot   v8.4s ,  v4.16b,  v0.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q5*/  \
  "sdot   v11.4s ,  v4.16b,  v0.4b[1]\n" /* out1 = b0 * a10[1], b0 = q5*/  \
  "sdot   v14.4s,  v4.16b,  v0.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q5*/  \
  "sdot   v17.4s,  v4.16b,  v0.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q5*/  \
  "sdot   v20.4s,  v4.16b,  v1.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q5*/  \
  "sdot   v23.4s,  v4.16b,  v1.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q5*/  \
  "sdot   v26.4s,  v4.16b,  v1.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q5*/  \
  "sdot   v29.4s,  v4.16b,  v1.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q5*/  \
  "sdot   v9.4s,  v5.16b,  v0.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q6*/  \
  "sdot   v12.4s,  v5.16b,  v0.4b[1]\n"  /* out9 = b1 * a10[1], b1 = q6*/  \
  "sdot   v15.4s,  v5.16b,  v0.4b[2]\n"  /* out10 = b1 * a10[2], b1 = q6*/ \
  "sdot   v18.4s,  v5.16b,  v0.4b[3]\n"  /* out11 = b1 * a10[3], b1 = q6*/ \
  "sdot   v21.4s,  v5.16b,  v1.4b[0]\n"  /* out12 = b1 * a10[0], b1 = q6*/ \
  "sdot   v24.4s,  v5.16b,  v1.4b[1]\n"  /* out13 = b1 * a10[1], b1 = q6*/ \
  "sdot   v27.4s,  v5.16b,  v1.4b[2]\n"  /* out14 = b1 * a10[2], b1 = q6*/ \
  "sdot   v30.4s,  v5.16b,  v1.4b[3]\n"  /* out15 = b1 * a10[3], b1 = q6*/ \
  "sdot   v10.4s,  v6.16b,  v0.4b[0]\n"  /* out16 = b2 * a10[0], b2 = q7*/ \
  "sdot   v13.4s,  v6.16b,  v0.4b[1]\n"  /* out17 = b2 * a10[0], b2 = q7*/ \
  "sdot   v16.4s,  v6.16b,  v0.4b[2]\n"  /* out18 = b2 * a10[0], b2 = q7*/ \
  "sdot   v19.4s,  v6.16b,  v0.4b[3]\n"  /* out19 = b2 * a10[0], b2 = q7*/ \
  "sdot   v22.4s,  v6.16b,  v1.4b[0]\n"  /* out20 = b2 * a10[0], b2 = q7*/ \
  "sdot   v25.4s,  v6.16b,  v1.4b[1]\n"  /* out21 = b2 * a10[0], b2 = q7*/ \
  "sdot   v28.4s,  v6.16b,  v1.4b[2]\n"  /* out22 = b2 * a10[0], b2 = q7*/ \
  "sdot   v31.4s,  v6.16b,  v1.4b[3]\n"  /* out23 = b2 * a10[0], b2 = q7*/ \
  "b      11f\n"                         /* tails==2 final tail*/          \
  "4:\n"                                 /* tail = 2*/                     \
  "sdot   v8.4s ,  v7.16b,  v2.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q5*/  \
  "sdot   v11.4s ,  v7.16b,  v2.4b[1]\n" /* out1 = b0 * a10[1], b0 = q5*/  \
  "sdot   v14.4s,  v7.16b,  v2.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q5*/  \
  "sdot   v17.4s,  v7.16b,  v2.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q5*/  \
  "sdot   v20.4s,  v7.16b,  v3.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q5*/  \
  "sdot   v23.4s,  v7.16b,  v3.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q5*/  \
  "sdot   v26.4s,  v7.16b,  v3.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q5*/  \
  "sdot   v29.4s,  v7.16b,  v3.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q5*/  \
  "sdot   v9.4s,  v4.16b,  v2.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q6*/  \
  "sdot   v12.4s,  v4.16b,  v2.4b[1]\n"  /* out9 = b1 * a10[1], b1 = q6*/  \
  "sdot   v15.4s,  v4.16b,  v2.4b[2]\n"  /* out10 = b1 * a10[2], b1 = q6*/ \
  "sdot   v18.4s,  v4.16b,  v2.4b[3]\n"  /* out11 = b1 * a10[3], b1 = q6*/ \
  "sdot   v21.4s,  v4.16b,  v3.4b[0]\n"  /* out12 = b1 * a10[0], b1 = q6*/ \
  "sdot   v24.4s,  v4.16b,  v3.4b[1]\n"  /* out13 = b1 * a10[1], b1 = q6*/ \
  "sdot   v27.4s,  v4.16b,  v3.4b[2]\n"  /* out14 = b1 * a10[2], b1 = q6*/ \
  "sdot   v30.4s,  v4.16b,  v3.4b[3]\n"  /* out15 = b1 * a10[3], b1 = q6*/ \
  "sdot   v10.4s,  v5.16b,  v2.4b[0]\n"  /* out16 = b2 * a10[0], b2 = q7*/ \
  "sdot   v13.4s,  v5.16b,  v2.4b[1]\n"  /* out17 = b2 * a10[0], b2 = q7*/ \
  "sdot   v16.4s,  v5.16b,  v2.4b[2]\n"  /* out18 = b2 * a10[0], b2 = q7*/ \
  "sdot   v19.4s,  v5.16b,  v2.4b[3]\n"  /* out19 = b2 * a10[0], b2 = q7*/ \
  "sdot   v22.4s,  v5.16b,  v3.4b[0]\n"  /* out20 = b2 * a10[0], b2 = q7*/ \
  "sdot   v25.4s,  v5.16b,  v3.4b[1]\n"  /* out21 = b2 * a10[0], b2 = q7*/ \
  "sdot   v28.4s,  v5.16b,  v3.4b[2]\n"  /* out22 = b2 * a10[0], b2 = q7*/ \
  "sdot   v31.4s,  v5.16b,  v3.4b[3]\n"  /* out23 = b2 * a10[0], b2 = q7*/ \
  "b      11f\n"                         /* tails==3 final tail*/          \
  "5:\n"                                 /* tail = 3*/                     \
  "ldr    q4, [%[b_ptr]], #16\n"         /* load b2, b0 to q4*/            \
  "sdot   v8.4s ,  v6.16b,  v0.4b[0]\n"  /* out0 = b0 * a10[0], b0 = q5*/  \
  "sdot   v11.4s ,  v6.16b,  v0.4b[1]\n" /* out1 = b0 * a10[1], b0 = q5*/  \
  "sdot   v14.4s,  v6.16b,  v0.4b[2]\n"  /* out2 = b0 * a10[2], b0 = q5*/  \
  "sdot   v17.4s,  v6.16b,  v0.4b[3]\n"  /* out3 = b0 * a10[3], b0 = q5*/  \
  "sdot   v20.4s,  v6.16b,  v1.4b[0]\n"  /* out4 = b0 * a11[0], b0 = q5*/  \
  "sdot   v23.4s,  v6.16b,  v1.4b[1]\n"  /* out5 = b0 * a11[1], b0 = q5*/  \
  "sdot   v26.4s,  v6.16b,  v1.4b[2]\n"  /* out6 = b0 * a11[2], b0 = q5*/  \
  "sdot   v29.4s,  v6.16b,  v1.4b[3]\n"  /* out7 = b0 * a11[3], b0 = q5*/  \
  "sdot   v9.4s,  v7.16b,  v0.4b[0]\n"   /* out8 = b0 * a10[0], b1 = q6*/  \
  "sdot   v12.4s,  v7.16b,  v0.4b[1]\n"  /* out9 = b1 * a10[1], b1 = q6*/  \
  "sdot   v15.4s,  v7.16b,  v0.4b[2]\n"  /* out10 = b1 * a10[2], b1 = q6*/ \
  "sdot   v18.4s,  v7.16b,  v0.4b[3]\n"  /* out11 = b1 * a10[3], b1 = q6*/ \
  "sdot   v21.4s,  v7.16b,  v1.4b[0]\n"  /* out12 = b1 * a10[0], b1 = q6*/ \
  "sdot   v24.4s,  v7.16b,  v1.4b[1]\n"  /* out13 = b1 * a10[1], b1 = q6*/ \
  "sdot   v27.4s,  v7.16b,  v1.4b[2]\n"  /* out14 = b1 * a10[2], b1 = q6*/ \
  "sdot   v30.4s,  v7.16b,  v1.4b[3]\n"  /* out15 = b1 * a10[3], b1 = q6*/ \
  "sdot   v10.4s,  v4.16b,  v0.4b[0]\n"  /* out16 = b2 * a10[0], b2 = q7*/ \
  "sdot   v13.4s,  v4.16b,  v0.4b[1]\n"  /* out17 = b2 * a10[0], b2 = q7*/ \
  "sdot   v16.4s,  v4.16b,  v0.4b[2]\n"  /* out18 = b2 * a10[0], b2 = q7*/ \
  "sdot   v19.4s,  v4.16b,  v0.4b[3]\n"  /* out19 = b2 * a10[0], b2 = q7*/ \
  "sdot   v22.4s,  v4.16b,  v1.4b[0]\n"  /* out20 = b2 * a10[0], b2 = q7*/ \
  "sdot   v25.4s,  v4.16b,  v1.4b[1]\n"  /* out21 = b2 * a10[0], b2 = q7*/ \
  "sdot   v28.4s,  v4.16b,  v1.4b[2]\n"  /* out22 = b2 * a10[0], b2 = q7*/ \
  "sdot   v31.4s,  v4.16b,  v1.4b[3]\n"  /* out23 = b2 * a10[0], b2 = q7*/ \
  "11: \n"                               /* end */

#define GEMM_SDOT_INT8_KERNEL_8x8     \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"  /* preload a*/                     \
  "eor    v8.16b,  v8.16b,  v8.16b \n"     /* out0 = 0 */                   \
  "eor    v11.16b, v11.16b, v11.16b\n"     /* out0 = 0 */                   \
  "eor    v14.16b, v14.16b, v14.16b\n"     /* out0 = 0 */                   \
  "eor    v17.16b, v17.16b, v17.16b\n"     /* out0 = 0 */                   \
  "eor    v20.16b, v20.16b, v20.16b\n"     /* out0 = 0 */                   \
  "eor    v23.16b, v23.16b, v23.16b\n"     /* out0 = 0 */                   \
  "eor    v26.16b, v26.16b, v26.16b\n"     /* out0 = 0 */                   \
  "eor    v29.16b, v29.16b, v29.16b\n"     /* out0 = 0 */                   \
  "prfm   pldl1keep, [%[b_ptr], #64]\n"  /* preload b*/                     \
  "eor    v9.16b,  v9.16b,  v9.16b \n"     /* out0 = 0 */                   \
  "eor    v12.16b, v12.16b, v12.16b\n"     /* out0 = 0 */                   \
  "eor    v15.16b, v15.16b, v15.16b\n"     /* out0 = 0 */                   \
  "eor    v18.16b, v18.16b, v18.16b\n"     /* out0 = 0 */                   \
  "eor    v21.16b, v21.16b, v21.16b\n"     /* out0 = 0 */                   \
  "eor    v24.16b, v24.16b, v24.16b\n"     /* out0 = 0 */                   \
  "eor    v27.16b, v27.16b, v27.16b\n"     /* out0 = 0 */                   \
  "eor    v30.16b, v30.16b, v30.16b\n"     /* out0 = 0 */                   \
  "1:\n"                                                                    \
  "ldp    q0, q1, [%[a_ptr]], #32\n"                                        \
  "ldp    q4, q5, [%[b_ptr]], #32\n"                                        \
  "sdot v8.4s, v4.16b, v0.4b[0]\n"                                          \
  "sdot v11.4s, v4.16b, v0.4b[1]\n"                                         \
  "sdot v14.4s, v4.16b, v0.4b[2]\n"                                         \
  "sdot v17.4s, v4.16b, v0.4b[3]\n"                                         \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"  /* preload a*/                     \
  "sdot v20.4s, v4.16b, v1.4b[0]\n"                                         \
  "sdot v23.4s, v4.16b, v1.4b[1]\n"                                         \
  "sdot v26.4s, v4.16b, v1.4b[2]\n"                                         \
  "sdot v29.4s, v4.16b, v1.4b[3]\n"                                         \
  "prfm   pldl1keep, [%[a_ptr], #128]\n"  /* preload b*/                    \
  "prfm   pldl1keep, [%[b_ptr], #64]\n"  /* preload b*/                     \
  "sdot v9.4s, v5.16b, v0.4b[0]\n"                                          \
  "sdot v12.4s, v5.16b, v0.4b[1]\n"                                         \
  "sdot v15.4s, v5.16b, v0.4b[2]\n"                                         \
  "sdot v18.4s, v5.16b, v0.4b[3]\n"                                         \
  "prfm   pldl1keep, [%[b_ptr], #128]\n"  /* preload b*/                    \
  "sdot v21.4s, v5.16b, v1.4b[0]\n"                                         \
  "sdot v24.4s, v5.16b, v1.4b[1]\n"                                         \
  "sdot v27.4s, v5.16b, v1.4b[2]\n"                                         \
  "sdot v30.4s, v5.16b, v1.4b[3]\n"                                         \
  "subs %w[k], %w[k], #1\n"                                                 \
  "bne 1b\n"

#define GEMM_SDOT_INT8_KERNEL_8x4     \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"  /* preload a*/                      \
  "eor    v8.16b,  v8.16b,  v8.16b \n"     /* out0 = 0 */                    \
  "eor    v11.16b, v11.16b, v11.16b\n"     /* out0 = 0 */                    \
  "eor    v14.16b, v14.16b, v14.16b\n"     /* out0 = 0 */                    \
  "eor    v17.16b, v17.16b, v17.16b\n"     /* out0 = 0 */                    \
  "prfm   pldl1keep, [%[b_ptr], #32]\n"  /* preload b*/                      \
  "eor    v20.16b, v20.16b, v20.16b\n"     /* out0 = 0 */                    \
  "eor    v23.16b, v23.16b, v23.16b\n"     /* out0 = 0 */                    \
  "eor    v26.16b, v26.16b, v26.16b\n"     /* out0 = 0 */                    \
  "eor    v29.16b, v29.16b, v29.16b\n"     /* out0 = 0 */                    \
  "1:\n"                              \
  "ldp    q0, q1, [%[a_ptr]], #32\n"  \
  "ldr    q4,  [%[b_ptr]], #16\n"     \
  "sdot v8.4s,  v4.16b, v0.4b[0]\n"    \
  "sdot v11.4s, v4.16b, v0.4b[1]\n"   \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"  /* preload a*/                      \
  "sdot v14.4s, v4.16b, v0.4b[2]\n"   \
  "sdot v17.4s, v4.16b, v0.4b[3]\n"   \
  "prfm   pldl1keep, [%[a_ptr], #64]\n"  /* preload a*/                      \
  "sdot v20.4s, v4.16b, v1.4b[0]\n"   \
  "sdot v23.4s, v4.16b, v1.4b[1]\n"   \
  "prfm   pldl1keep, [%[b_ptr], #32]\n"  /* preload b*/                      \
  "sdot v26.4s, v4.16b, v1.4b[2]\n"   \
  "sdot v29.4s, v4.16b, v1.4b[3]\n"   \
  "subs %w[k], %w[k], #1\n"           \
  "bne 1b\n"
