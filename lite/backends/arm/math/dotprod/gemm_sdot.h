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
".word 0x4f80e088\n" /* sdot v8.4s, v4.16b, v0.4b[0] */\
".word 0x4fa0e08b\n" /* sdot v11.4s, v4.16b, v0.4b[1] */\
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b2, b0 to q6, q7       */ \
".word 0x4f80e88e\n" /* sdot v14.4s, v4.16b, v0.4b[2] */\
".word 0x4fa0e891\n" /* sdot v17.4s, v4.16b, v0.4b[3] */\
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q3, q4     */ \
".word 0x4f81e094\n" /* sdot v20.4s, v4.16b, v1.4b[0] */\
".word 0x4fa1e097\n" /* sdot v23.4s, v4.16b, v1.4b[1] */\
".word 0x4f81e89a\n" /* sdot v26.4s, v4.16b, v1.4b[2] */\
".word 0x4fa1e89d\n" /* sdot v29.4s, v4.16b, v1.4b[3] */\
".word 0x4f80e0a9\n" /* sdot v9.4s, v5.16b, v0.4b[0] */\
".word 0x4fa0e0ac\n" /* sdot v12.4s, v5.16b, v0.4b[1] */\
".word 0x4f80e8af\n" /* sdot v15.4s, v5.16b, v0.4b[2] */\
".word 0x4fa0e8b2\n" /* sdot v18.4s, v5.16b, v0.4b[3] */\
".word 0x4f81e0b5\n" /* sdot v21.4s, v5.16b, v1.4b[0] */\
".word 0x4fa1e0b8\n" /* sdot v24.4s, v5.16b, v1.4b[1] */\
".word 0x4f81e8bb\n" /* sdot v27.4s, v5.16b, v1.4b[2] */\
".word 0x4fa1e8be\n" /* sdot v30.4s, v5.16b, v1.4b[3] */\
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b1, b2 to q4, q5       */ \
".word 0x4f80e0ca\n" /* sdot v10.4s, v6.16b, v0.4b[0] */\
".word 0x4fa0e0cd\n" /* sdot v13.4s, v6.16b, v0.4b[1] */\
  "prfm   pldl1keep, [%[b_ptr], #384]\n"                                   \
".word 0x4f80e8d0\n" /* sdot v16.4s, v6.16b, v0.4b[2] */\
".word 0x4fa0e8d3\n" /* sdot v19.4s, v6.16b, v0.4b[3] */\
".word 0x4f81e0d6\n" /* sdot v22.4s, v6.16b, v1.4b[0] */\
".word 0x4fa1e0d9\n" /* sdot v25.4s, v6.16b, v1.4b[1] */\
".word 0x4f81e8dc\n" /* sdot v28.4s, v6.16b, v1.4b[2] */\
".word 0x4fa1e8df\n" /* sdot v31.4s, v6.16b, v1.4b[3] */\
  "ldp    q0, q1, [%[a_ptr]], #32\n"    /* load a00, a01 to q0, q1 */      \
  /* unrool 1 */                                                           \
".word 0x4f82e0e8\n" /* sdot v8.4s, v7.16b, v2.4b[0] */\
".word 0x4fa2e0eb\n" /* sdot v11.4s, v7.16b, v2.4b[1] */\
".word 0x4f82e8ee\n" /* sdot v14.4s, v7.16b, v2.4b[2] */\
  "prfm   pldl1keep, [%[a_ptr], #256]\n"                                   \
".word 0x4fa2e8f1\n" /* sdot v17.4s, v7.16b, v2.4b[3] */\
".word 0x4f83e0f4\n" /* sdot v20.4s, v7.16b, v3.4b[0] */\
".word 0x4fa3e0f7\n" /* sdot v23.4s, v7.16b, v3.4b[1] */\
".word 0x4f83e8fa\n" /* sdot v26.4s, v7.16b, v3.4b[2] */\
".word 0x4fa3e8fd\n" /* sdot v29.4s, v7.16b, v3.4b[3] */\
  "ldp    q6, q7, [%[b_ptr]], #32\n"    /* load b0, b1 to q6, q7       */  \
".word 0x4f82e089\n" /* sdot v9.4s, v4.16b, v2.4b[0] */\
".word 0x4fa2e08c\n" /* sdot v12.4s, v4.16b, v2.4b[1] */\
".word 0x4f82e88f\n" /* sdot v15.4s, v4.16b, v2.4b[2] */\
".word 0x4fa2e892\n" /* sdot v18.4s, v4.16b, v2.4b[3] */\
".word 0x4f83e095\n" /* sdot v21.4s, v4.16b, v3.4b[0] */\
".word 0x4fa3e098\n" /* sdot v24.4s, v4.16b, v3.4b[1] */\
".word 0x4f83e89b\n" /* sdot v27.4s, v4.16b, v3.4b[2] */\
".word 0x4fa3e89e\n" /* sdot v30.4s, v4.16b, v3.4b[3] */\
".word 0x4f82e0aa\n" /* sdot v10.4s, v5.16b, v2.4b[0] */\
".word 0x4fa2e0ad\n" /* sdot v13.4s, v5.16b, v2.4b[1] */\
".word 0x4f82e8b0\n" /* sdot v16.4s, v5.16b, v2.4b[2] */\
".word 0x4fa2e8b3\n" /* sdot v19.4s, v5.16b, v2.4b[3] */\
".word 0x4f83e0b6\n" /* sdot v22.4s, v5.16b, v3.4b[0] */\
".word 0x4fa3e0b9\n" /* sdot v25.4s, v5.16b, v3.4b[1] */\
".word 0x4f83e8bc\n" /* sdot v28.4s, v5.16b, v3.4b[2] */\
".word 0x4fa3e8bf\n" /* sdot v31.4s, v5.16b, v3.4b[3] */\
  "ldp    q4, q5, [%[b_ptr]], #32\n"    /* load b2, b0 to q4, q5 */        \
  /* unrool 2*/                                                            \
".word 0x4f80e0c8\n" /* sdot v8.4s, v6.16b, v0.4b[0] */\
".word 0x4fa0e0cb\n" /* sdot v11.4s, v6.16b, v0.4b[1] */\
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q3, q4*/      \
".word 0x4f80e8ce\n" /* sdot v14.4s, v6.16b, v0.4b[2] */\
".word 0x4fa0e8d1\n" /* sdot v17.4s, v6.16b, v0.4b[3] */\
".word 0x4f81e0d4\n" /* sdot v20.4s, v6.16b, v1.4b[0] */\
".word 0x4fa1e0d7\n" /* sdot v23.4s, v6.16b, v1.4b[1] */\
".word 0x4f81e8da\n" /* sdot v26.4s, v6.16b, v1.4b[2] */\
".word 0x4fa1e8dd\n" /* sdot v29.4s, v6.16b, v1.4b[3] */\
".word 0x4f80e0e9\n" /* sdot v9.4s, v7.16b, v0.4b[0] */\
".word 0x4fa0e0ec\n" /* sdot v12.4s, v7.16b, v0.4b[1] */\
  "prfm   pldl1keep, [%[b_ptr], #384]\n"                                   \
".word 0x4f80e8ef\n" /* sdot v15.4s, v7.16b, v0.4b[2] */\
".word 0x4fa0e8f2\n" /* sdot v18.4s, v7.16b, v0.4b[3] */\
".word 0x4f81e0f5\n" /* sdot v21.4s, v7.16b, v1.4b[0] */\
".word 0x4fa1e0f8\n" /* sdot v24.4s, v7.16b, v1.4b[1] */\
".word 0x4f81e8fb\n" /* sdot v27.4s, v7.16b, v1.4b[2] */\
".word 0x4fa1e8fe\n" /* sdot v30.4s, v7.16b, v1.4b[3] */\
  "ldp    q6, q7, [%[b_ptr]], #32\n"    /* load b1, b2 to q6, q7*/         \
".word 0x4f80e08a\n" /* sdot v10.4s, v4.16b, v0.4b[0] */\
".word 0x4fa0e08d\n" /* sdot v13.4s, v4.16b, v0.4b[1] */\
".word 0x4f80e890\n" /* sdot v16.4s, v4.16b, v0.4b[2] */\
".word 0x4fa0e893\n" /* sdot v19.4s, v4.16b, v0.4b[3] */\
".word 0x4f81e096\n" /* sdot v22.4s, v4.16b, v1.4b[0] */\
".word 0x4fa1e099\n" /* sdot v25.4s, v4.16b, v1.4b[1] */\
".word 0x4f81e89c\n" /* sdot v28.4s, v4.16b, v1.4b[2] */\
".word 0x4fa1e89f\n" /* sdot v31.4s, v4.16b, v1.4b[3] */\
  "ldp    q0, q1, [%[a_ptr]], #32\n" /* load a00, a01 to q0, q1*/          \
  /* unrool 3*/                                                            \
".word 0x4f82e0a8\n" /* sdot v8.4s, v5.16b, v2.4b[0] */\
".word 0x4fa2e0ab\n" /* sdot v11.4s, v5.16b, v2.4b[1] */\
".word 0x4f82e8ae\n" /* sdot v14.4s, v5.16b, v2.4b[2] */\
".word 0x4fa2e8b1\n" /* sdot v17.4s, v5.16b, v2.4b[3] */\
".word 0x4f83e0b4\n" /* sdot v20.4s, v5.16b, v3.4b[0] */\
".word 0x4fa3e0b7\n" /* sdot v23.4s, v5.16b, v3.4b[1] */\
".word 0x4f83e8ba\n" /* sdot v26.4s, v5.16b, v3.4b[2] */\
".word 0x4fa3e8bd\n" /* sdot v29.4s, v5.16b, v3.4b[3] */\
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b0, b1 to q4, q5*/        \
".word 0x4f82e0c9\n" /* sdot v9.4s, v6.16b, v2.4b[0] */\
".word 0x4fa2e0cc\n" /* sdot v12.4s, v6.16b, v2.4b[1] */\
  "prfm   pldl1keep, [%[a_ptr], #256]\n"                                   \
".word 0x4f82e8cf\n" /* sdot v15.4s, v6.16b, v2.4b[2] */\
".word 0x4fa2e8d2\n" /* sdot v18.4s, v6.16b, v2.4b[3] */\
".word 0x4f83e0d5\n" /* sdot v21.4s, v6.16b, v3.4b[0] */\
".word 0x4fa3e0d8\n" /* sdot v24.4s, v6.16b, v3.4b[1] */\
".word 0x4f83e8db\n" /* sdot v27.4s, v6.16b, v3.4b[2] */\
  "prfm   pldl1keep, [%[b_ptr], #384]\n"                                   \
".word 0x4fa3e8de\n" /* sdot v30.4s, v6.16b, v3.4b[3] */\
".word 0x4f82e0ea\n" /* sdot v10.4s, v7.16b, v2.4b[0] */\
".word 0x4fa2e0ed\n" /* sdot v13.4s, v7.16b, v2.4b[1] */\
".word 0x4f82e8f0\n" /* sdot v16.4s, v7.16b, v2.4b[2] */\
".word 0x4fa2e8f3\n" /* sdot v19.4s, v7.16b, v2.4b[3] */\
".word 0x4f83e0f6\n" /* sdot v22.4s, v7.16b, v3.4b[0] */\
".word 0x4fa3e0f9\n" /* sdot v25.4s, v7.16b, v3.4b[1] */\
  "subs   %w[k], %w[k], #1\n"           /* loop count - 1*/                \
".word 0x4f83e8fc\n" /* sdot v28.4s, v7.16b, v3.4b[2] */\
".word 0x4fa3e8ff\n" /* sdot v31.4s, v7.16b, v3.4b[3] */\
  "bne    1b\n" /* Target to use when K is 1 or 2 */                       \
  "2:\n"                                             /* process tail*/     \
  "subs       %w[tail], %w[tail], #1\n"              /* tail--*/           \
  "beq        3f\n" /*jump to tail = 1*/                                   \
  /* final unrool 0, unrool 0, tail > 1*/                                  \
".word 0x4f80e088\n" /* sdot v8.4s, v4.16b, v0.4b[0] */\
".word 0x4fa0e08b\n" /* sdot v11.4s, v4.16b, v0.4b[1] */\
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b2, b0 to q6, q7*/        \
".word 0x4f80e88e\n" /* sdot v14.4s, v4.16b, v0.4b[2] */\
".word 0x4fa0e891\n" /* sdot v17.4s, v4.16b, v0.4b[3] */\
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q2, q3*/      \
".word 0x4f81e094\n" /* sdot v20.4s, v4.16b, v1.4b[0] */\
".word 0x4fa1e097\n" /* sdot v23.4s, v4.16b, v1.4b[1] */\
".word 0x4f81e89a\n" /* sdot v26.4s, v4.16b, v1.4b[2] */\
".word 0x4fa1e89d\n" /* sdot v29.4s, v4.16b, v1.4b[3] */\
  "subs   %w[tail], %w[tail], #1\n"      /* tail--*/                       \
".word 0x4f80e0a9\n" /* sdot v9.4s, v5.16b, v0.4b[0] */\
".word 0x4fa0e0ac\n" /* sdot v12.4s, v5.16b, v0.4b[1] */\
".word 0x4f80e8af\n" /* sdot v15.4s, v5.16b, v0.4b[2] */\
".word 0x4fa0e8b2\n" /* sdot v18.4s, v5.16b, v0.4b[3] */\
".word 0x4f81e0b5\n" /* sdot v21.4s, v5.16b, v1.4b[0] */\
".word 0x4fa1e0b8\n" /* sdot v24.4s, v5.16b, v1.4b[1] */\
".word 0x4f81e8bb\n" /* sdot v27.4s, v5.16b, v1.4b[2] */\
".word 0x4fa1e8be\n" /* sdot v30.4s, v5.16b, v1.4b[3] */\
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b1, b2 to q4, q5*/        \
".word 0x4f80e0ca\n" /* sdot v10.4s, v6.16b, v0.4b[0] */\
".word 0x4fa0e0cd\n" /* sdot v13.4s, v6.16b, v0.4b[1] */\
".word 0x4f80e8d0\n" /* sdot v16.4s, v6.16b, v0.4b[2] */\
".word 0x4fa0e8d3\n" /* sdot v19.4s, v6.16b, v0.4b[3] */\
".word 0x4f81e0d6\n" /* sdot v22.4s, v6.16b, v1.4b[0] */\
".word 0x4fa1e0d9\n" /* sdot v25.4s, v6.16b, v1.4b[1] */\
".word 0x4f81e8dc\n" /* sdot v28.4s, v6.16b, v1.4b[2] */\
".word 0x4fa1e8df\n" /* sdot v31.4s, v6.16b, v1.4b[3] */\
  "beq        4f\n" /*jump to tail = 2*/                                   \
  /* unrool 1, tail > 2*/                                                  \
  "ldp    q0, q1, [%[a_ptr]], #32\n"     /* load a00, a01 to q0, q1*/      \
".word 0x4f82e0e8\n" /* sdot v8.4s, v7.16b, v2.4b[0] */\
".word 0x4fa2e0eb\n" /* sdot v11.4s, v7.16b, v2.4b[1] */\
".word 0x4f82e8ee\n" /* sdot v14.4s, v7.16b, v2.4b[2] */\
".word 0x4fa2e8f1\n" /* sdot v17.4s, v7.16b, v2.4b[3] */\
".word 0x4f83e0f4\n" /* sdot v20.4s, v7.16b, v3.4b[0] */\
".word 0x4fa3e0f7\n" /* sdot v23.4s, v7.16b, v3.4b[1] */\
".word 0x4f83e8fa\n" /* sdot v26.4s, v7.16b, v3.4b[2] */\
".word 0x4fa3e8fd\n" /* sdot v29.4s, v7.16b, v3.4b[3] */\
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b0, b1 to q6, q7*/        \
".word 0x4f82e089\n" /* sdot v9.4s, v4.16b, v2.4b[0] */\
".word 0x4fa2e08c\n" /* sdot v12.4s, v4.16b, v2.4b[1] */\
".word 0x4f82e88f\n" /* sdot v15.4s, v4.16b, v2.4b[2] */\
".word 0x4fa2e892\n" /* sdot v18.4s, v4.16b, v2.4b[3] */\
".word 0x4f83e095\n" /* sdot v21.4s, v4.16b, v3.4b[0] */\
".word 0x4fa3e098\n" /* sdot v24.4s, v4.16b, v3.4b[1] */\
".word 0x4f83e89b\n" /* sdot v27.4s, v4.16b, v3.4b[2] */\
".word 0x4fa3e89e\n" /* sdot v30.4s, v4.16b, v3.4b[3] */\
  "subs   %w[tail], %w[tail], #1\n"      /* tail--*/                       \
".word 0x4f82e0aa\n" /* sdot v10.4s, v5.16b, v2.4b[0] */\
".word 0x4fa2e0ad\n" /* sdot v13.4s, v5.16b, v2.4b[1] */\
".word 0x4f82e8b0\n" /* sdot v16.4s, v5.16b, v2.4b[2] */\
".word 0x4fa2e8b3\n" /* sdot v19.4s, v5.16b, v2.4b[3] */\
".word 0x4f83e0b6\n" /* sdot v22.4s, v5.16b, v3.4b[0] */\
".word 0x4fa3e0b9\n" /* sdot v25.4s, v5.16b, v3.4b[1] */\
".word 0x4f83e8bc\n" /* sdot v28.4s, v5.16b, v3.4b[2] */\
".word 0x4fa3e8bf\n" /* sdot v31.4s, v5.16b, v3.4b[3] */\
  "beq        5f\n" /*jump to tail = 3*/                                   \
  /* unrool 2, tail = 4*/                                                  \
  "ldp    q4, q5, [%[b_ptr]], #32\n"     /* load b2, b0 to q4, q5*/        \
".word 0x4f80e0c8\n" /* sdot v8.4s, v6.16b, v0.4b[0] */\
".word 0x4fa0e0cb\n" /* sdot v11.4s, v6.16b, v0.4b[1] */\
  "ldp    q2, q3, [%[a_ptr]], #32\n"     /* load a10, a11 to q3, q4*/      \
".word 0x4f80e8ce\n" /* sdot v14.4s, v6.16b, v0.4b[2] */\
".word 0x4fa0e8d1\n" /* sdot v17.4s, v6.16b, v0.4b[3] */\
".word 0x4f81e0d4\n" /* sdot v20.4s, v6.16b, v1.4b[0] */\
".word 0x4fa1e0d7\n" /* sdot v23.4s, v6.16b, v1.4b[1] */\
".word 0x4f81e8da\n" /* sdot v26.4s, v6.16b, v1.4b[2] */\
".word 0x4fa1e8dd\n" /* sdot v29.4s, v6.16b, v1.4b[3] */\
".word 0x4f80e0e9\n" /* sdot v9.4s, v7.16b, v0.4b[0] */\
".word 0x4fa0e0ec\n" /* sdot v12.4s, v7.16b, v0.4b[1] */\
".word 0x4f80e8ef\n" /* sdot v15.4s, v7.16b, v0.4b[2] */\
".word 0x4fa0e8f2\n" /* sdot v18.4s, v7.16b, v0.4b[3] */\
".word 0x4f81e0f5\n" /* sdot v21.4s, v7.16b, v1.4b[0] */\
".word 0x4fa1e0f8\n" /* sdot v24.4s, v7.16b, v1.4b[1] */\
".word 0x4f81e8fb\n" /* sdot v27.4s, v7.16b, v1.4b[2] */\
".word 0x4fa1e8fe\n" /* sdot v30.4s, v7.16b, v1.4b[3] */\
  "ldp    q6, q7, [%[b_ptr]], #32\n"     /* load b1, b2 to q6, q7*/        \
".word 0x4f80e08a\n" /* sdot v10.4s, v4.16b, v0.4b[0] */\
".word 0x4fa0e08d\n" /* sdot v13.4s, v4.16b, v0.4b[1] */\
".word 0x4f80e890\n" /* sdot v16.4s, v4.16b, v0.4b[2] */\
".word 0x4fa0e893\n" /* sdot v19.4s, v4.16b, v0.4b[3] */\
".word 0x4f81e096\n" /* sdot v22.4s, v4.16b, v1.4b[0] */\
".word 0x4fa1e099\n" /* sdot v25.4s, v4.16b, v1.4b[1] */\
".word 0x4f81e89c\n" /* sdot v28.4s, v4.16b, v1.4b[2] */\
".word 0x4fa1e89f\n" /* sdot v31.4s, v4.16b, v1.4b[3] */\
  /* unrool 3, tail = 4*/                                                  \
".word 0x4f82e0a8\n" /* sdot v8.4s, v5.16b, v2.4b[0] */\
".word 0x4fa2e0ab\n" /* sdot v11.4s, v5.16b, v2.4b[1] */\
".word 0x4f82e8ae\n" /* sdot v14.4s, v5.16b, v2.4b[2] */\
".word 0x4fa2e8b1\n" /* sdot v17.4s, v5.16b, v2.4b[3] */\
".word 0x4f83e0b4\n" /* sdot v20.4s, v5.16b, v3.4b[0] */\
".word 0x4fa3e0b7\n" /* sdot v23.4s, v5.16b, v3.4b[1] */\
".word 0x4f83e8ba\n" /* sdot v26.4s, v5.16b, v3.4b[2] */\
".word 0x4fa3e8bd\n" /* sdot v29.4s, v5.16b, v3.4b[3] */\
".word 0x4f82e0c9\n" /* sdot v9.4s, v6.16b, v2.4b[0] */\
".word 0x4fa2e0cc\n" /* sdot v12.4s, v6.16b, v2.4b[1] */\
".word 0x4f82e8cf\n" /* sdot v15.4s, v6.16b, v2.4b[2] */\
".word 0x4fa2e8d2\n" /* sdot v18.4s, v6.16b, v2.4b[3] */\
".word 0x4f83e0d5\n" /* sdot v21.4s, v6.16b, v3.4b[0] */\
".word 0x4fa3e0d8\n" /* sdot v24.4s, v6.16b, v3.4b[1] */\
".word 0x4f83e8db\n" /* sdot v27.4s, v6.16b, v3.4b[2] */\
".word 0x4fa3e8de\n" /* sdot v30.4s, v6.16b, v3.4b[3] */\
".word 0x4f82e0ea\n" /* sdot v10.4s, v7.16b, v2.4b[0] */\
".word 0x4fa2e0ed\n" /* sdot v13.4s, v7.16b, v2.4b[1] */\
".word 0x4f82e8f0\n" /* sdot v16.4s, v7.16b, v2.4b[2] */\
".word 0x4fa2e8f3\n" /* sdot v19.4s, v7.16b, v2.4b[3] */\
".word 0x4f83e0f6\n" /* sdot v22.4s, v7.16b, v3.4b[0] */\
".word 0x4fa3e0f9\n" /* sdot v25.4s, v7.16b, v3.4b[1] */\
".word 0x4f83e8fc\n" /* sdot v28.4s, v7.16b, v3.4b[2] */\
".word 0x4fa3e8ff\n" /* sdot v31.4s, v7.16b, v3.4b[3] */\
  "b      11f\n"                         /* tails==1 final tail*/          \
  "3: \n"                                /* tail=1*/                       \
  "ldr    q6, [%[b_ptr]], #16\n"         /* load b2 to q6*/                \
".word 0x4f80e088\n" /* sdot v8.4s, v4.16b, v0.4b[0] */\
".word 0x4fa0e08b\n" /* sdot v11.4s, v4.16b, v0.4b[1] */\
".word 0x4f80e88e\n" /* sdot v14.4s, v4.16b, v0.4b[2] */\
".word 0x4fa0e891\n" /* sdot v17.4s, v4.16b, v0.4b[3] */\
".word 0x4f81e094\n" /* sdot v20.4s, v4.16b, v1.4b[0] */\
".word 0x4fa1e097\n" /* sdot v23.4s, v4.16b, v1.4b[1] */\
".word 0x4f81e89a\n" /* sdot v26.4s, v4.16b, v1.4b[2] */\
".word 0x4fa1e89d\n" /* sdot v29.4s, v4.16b, v1.4b[3] */\
".word 0x4f80e0a9\n" /* sdot v9.4s, v5.16b, v0.4b[0] */\
".word 0x4fa0e0ac\n" /* sdot v12.4s, v5.16b, v0.4b[1] */\
".word 0x4f80e8af\n" /* sdot v15.4s, v5.16b, v0.4b[2] */\
".word 0x4fa0e8b2\n" /* sdot v18.4s, v5.16b, v0.4b[3] */\
".word 0x4f81e0b5\n" /* sdot v21.4s, v5.16b, v1.4b[0] */\
".word 0x4fa1e0b8\n" /* sdot v24.4s, v5.16b, v1.4b[1] */\
".word 0x4f81e8bb\n" /* sdot v27.4s, v5.16b, v1.4b[2] */\
".word 0x4fa1e8be\n" /* sdot v30.4s, v5.16b, v1.4b[3] */\
".word 0x4f80e0ca\n" /* sdot v10.4s, v6.16b, v0.4b[0] */\
".word 0x4fa0e0cd\n" /* sdot v13.4s, v6.16b, v0.4b[1] */\
".word 0x4f80e8d0\n" /* sdot v16.4s, v6.16b, v0.4b[2] */\
".word 0x4fa0e8d3\n" /* sdot v19.4s, v6.16b, v0.4b[3] */\
".word 0x4f81e0d6\n" /* sdot v22.4s, v6.16b, v1.4b[0] */\
".word 0x4fa1e0d9\n" /* sdot v25.4s, v6.16b, v1.4b[1] */\
".word 0x4f81e8dc\n" /* sdot v28.4s, v6.16b, v1.4b[2] */\
".word 0x4fa1e8df\n" /* sdot v31.4s, v6.16b, v1.4b[3] */\
  "b      11f\n"                         /* tails==2 final tail*/          \
  "4:\n"                                 /* tail = 2*/                     \
".word 0x4f82e0e8\n" /* sdot v8.4s, v7.16b, v2.4b[0] */\
".word 0x4fa2e0eb\n" /* sdot v11.4s, v7.16b, v2.4b[1] */\
".word 0x4f82e8ee\n" /* sdot v14.4s, v7.16b, v2.4b[2] */\
".word 0x4fa2e8f1\n" /* sdot v17.4s, v7.16b, v2.4b[3] */\
".word 0x4f83e0f4\n" /* sdot v20.4s, v7.16b, v3.4b[0] */\
".word 0x4fa3e0f7\n" /* sdot v23.4s, v7.16b, v3.4b[1] */\
".word 0x4f83e8fa\n" /* sdot v26.4s, v7.16b, v3.4b[2] */\
".word 0x4fa3e8fd\n" /* sdot v29.4s, v7.16b, v3.4b[3] */\
".word 0x4f82e089\n" /* sdot v9.4s, v4.16b, v2.4b[0] */\
".word 0x4fa2e08c\n" /* sdot v12.4s, v4.16b, v2.4b[1] */\
".word 0x4f82e88f\n" /* sdot v15.4s, v4.16b, v2.4b[2] */\
".word 0x4fa2e892\n" /* sdot v18.4s, v4.16b, v2.4b[3] */\
".word 0x4f83e095\n" /* sdot v21.4s, v4.16b, v3.4b[0] */\
".word 0x4fa3e098\n" /* sdot v24.4s, v4.16b, v3.4b[1] */\
".word 0x4f83e89b\n" /* sdot v27.4s, v4.16b, v3.4b[2] */\
".word 0x4fa3e89e\n" /* sdot v30.4s, v4.16b, v3.4b[3] */\
".word 0x4f82e0aa\n" /* sdot v10.4s, v5.16b, v2.4b[0] */\
".word 0x4fa2e0ad\n" /* sdot v13.4s, v5.16b, v2.4b[1] */\
".word 0x4f82e8b0\n" /* sdot v16.4s, v5.16b, v2.4b[2] */\
".word 0x4fa2e8b3\n" /* sdot v19.4s, v5.16b, v2.4b[3] */\
".word 0x4f83e0b6\n" /* sdot v22.4s, v5.16b, v3.4b[0] */\
".word 0x4fa3e0b9\n" /* sdot v25.4s, v5.16b, v3.4b[1] */\
".word 0x4f83e8bc\n" /* sdot v28.4s, v5.16b, v3.4b[2] */\
".word 0x4fa3e8bf\n" /* sdot v31.4s, v5.16b, v3.4b[3] */\
  "b      11f\n"                         /* tails==3 final tail*/          \
  "5:\n"                                 /* tail = 3*/                     \
  "ldr    q4, [%[b_ptr]], #16\n"         /* load b2, b0 to q4*/            \
".word 0x4f80e0c8\n" /* sdot v8.4s, v6.16b, v0.4b[0] */\
".word 0x4fa0e0cb\n" /* sdot v11.4s, v6.16b, v0.4b[1] */\
".word 0x4f80e8ce\n" /* sdot v14.4s, v6.16b, v0.4b[2] */\
".word 0x4fa0e8d1\n" /* sdot v17.4s, v6.16b, v0.4b[3] */\
".word 0x4f81e0d4\n" /* sdot v20.4s, v6.16b, v1.4b[0] */\
".word 0x4fa1e0d7\n" /* sdot v23.4s, v6.16b, v1.4b[1] */\
".word 0x4f81e8da\n" /* sdot v26.4s, v6.16b, v1.4b[2] */\
".word 0x4fa1e8dd\n" /* sdot v29.4s, v6.16b, v1.4b[3] */\
".word 0x4f80e0e9\n" /* sdot v9.4s, v7.16b, v0.4b[0] */\
".word 0x4fa0e0ec\n" /* sdot v12.4s, v7.16b, v0.4b[1] */\
".word 0x4f80e8ef\n" /* sdot v15.4s, v7.16b, v0.4b[2] */\
".word 0x4fa0e8f2\n" /* sdot v18.4s, v7.16b, v0.4b[3] */\
".word 0x4f81e0f5\n" /* sdot v21.4s, v7.16b, v1.4b[0] */\
".word 0x4fa1e0f8\n" /* sdot v24.4s, v7.16b, v1.4b[1] */\
".word 0x4f81e8fb\n" /* sdot v27.4s, v7.16b, v1.4b[2] */\
".word 0x4fa1e8fe\n" /* sdot v30.4s, v7.16b, v1.4b[3] */\
".word 0x4f80e08a\n" /* sdot v10.4s, v4.16b, v0.4b[0] */\
".word 0x4fa0e08d\n" /* sdot v13.4s, v4.16b, v0.4b[1] */\
".word 0x4f80e890\n" /* sdot v16.4s, v4.16b, v0.4b[2] */\
".word 0x4fa0e893\n" /* sdot v19.4s, v4.16b, v0.4b[3] */\
".word 0x4f81e096\n" /* sdot v22.4s, v4.16b, v1.4b[0] */\
".word 0x4fa1e099\n" /* sdot v25.4s, v4.16b, v1.4b[1] */\
".word 0x4f81e89c\n" /* sdot v28.4s, v4.16b, v1.4b[2] */\
".word 0x4fa1e89f\n" /* sdot v31.4s, v4.16b, v1.4b[3] */\
  "11: \n"                               /* end */
