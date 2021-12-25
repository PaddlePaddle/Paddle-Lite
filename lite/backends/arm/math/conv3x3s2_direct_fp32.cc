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

#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {

const int OUT_C_BLOCK = 4;
const int OUT_H_BLOCK = 2;
const int OUT_W_BLOCK = 4;

size_t conv3x3s2_direct_workspace_size(const operators::ConvParam& param,
                                       ARMContext* ctx) {
  auto dim_in = param.x->dims();
  auto dim_out = param.output->dims();
  auto paddings = *param.paddings;
  const int threads = ctx->threads();
  int llc_size = ctx->llc_size() / sizeof(float);
  const int pad_w = paddings[2];
  const int pad_h = paddings[0];
  int ow = dim_out[3];
  int oh = dim_out[2];
  int ic = dim_in[1];
  if (ic == 3) {
    ic = 4;
  }
  const int wout_round = ROUNDUP(ow, OUT_W_BLOCK);
  const int win_round = wout_round * 2 /*stride_w*/ + 1;
  const int hin_r_block = OUT_H_BLOCK * 2 /*stride_h*/ + 1;

  int hout_r_block =
      (llc_size - 2 * wout_round * ic - ic) /
      ((4 * wout_round + 2) * ic + wout_round * OUT_C_BLOCK * threads);
  hout_r_block = hout_r_block > oh ? oh : hout_r_block;
  hout_r_block = (hout_r_block / OUT_H_BLOCK) * OUT_H_BLOCK;
  hout_r_block = hout_r_block < OUT_H_BLOCK ? OUT_H_BLOCK : hout_r_block;

  int in_len = win_round * ic;
  int pre_in_size = hin_r_block * in_len;
  int pre_out_size = OUT_C_BLOCK * hout_r_block * wout_round;

  return sizeof(float) * (pre_in_size + ctx->threads() * pre_out_size);
}

#ifdef __aarch64__
#define INIT_FIRST                       \
  "ldp    q0, q1,   [%[r0]], #32\n"      \
  "ldr    d10,      [%[r0]]\n"           \
  "ldp    q4, q5,   [%[r2]], #32\n"      \
  "ldr    d12,      [%[r2]]\n"           \
  "2:\n"                                 \
  "fmul   v15.4s,  %[w0].4s,  v0.s[0]\n" \
  "fmul   v16.4s,  %[w0].4s,  v0.s[2]\n" \
  "fmul   v17.4s,  %[w0].4s,  v1.s[0]\n" \
  "fmul   v18.4s,  %[w0].4s,  v1.s[2]\n" \
  "fmul   v19.4s,  %[w0].4s,  v4.s[0]\n" \
  "fmul   v20.4s,  %[w0].4s,  v4.s[2]\n" \
  "fmul   v21.4s,  %[w0].4s,  v5.s[0]\n" \
  "fmul   v22.4s,  %[w0].4s,  v5.s[2]\n"

#define INIT                              \
  "ldp    q15, q16, [%[ptr_out0]]\n"      \
  "ldp    q17, q18, [%[ptr_out0], #32]\n" \
  "ldp    q0, q1,   [%[r0]], #32\n"       \
  "ldr    d10,      [%[r0]]\n"            \
  "ldp    q4, q5,   [%[r2]], #32\n"       \
  "ldr    d12,      [%[r2]]\n"            \
  "2:\n"                                  \
  "ldp    q19, q20, [%[ptr_out1]]     \n" \
  "ldp    q21, q22, [%[ptr_out1], #32]\n" \
  "fmla   v15.4s,  %[w0].4s,  v0.s[0]\n"  \
  "fmla   v16.4s,  %[w0].4s,  v0.s[2]\n"  \
  "fmla   v17.4s,  %[w0].4s,  v1.s[0]\n"  \
  "fmla   v18.4s,  %[w0].4s,  v1.s[2]\n"  \
  "fmla   v19.4s,  %[w0].4s,  v4.s[0]\n"  \
  "fmla   v20.4s,  %[w0].4s,  v4.s[2]\n"  \
  "fmla   v21.4s,  %[w0].4s,  v5.s[0]\n"  \
  "fmla   v22.4s,  %[w0].4s,  v5.s[2]\n"

#define COMPUTE                           \
  "ldp    q2, q3,   [%[r1]], #32      \n" \
  "fmla   v15.4s,  %[w6].4s,  v4.s[0]\n"  \
  "fmla   v16.4s,  %[w6].4s,  v4.s[2]\n"  \
  "fmla   v17.4s,  %[w6].4s,  v5.s[0]\n"  \
  "fmla   v18.4s,  %[w6].4s,  v5.s[2]\n"  \
  "ldr    d11,      [%[r1]]\n"            \
  "fmla   v15.4s,  %[w1].4s,  v0.s[1]\n"  \
  "fmla   v16.4s,  %[w1].4s,  v0.s[3]\n"  \
  "fmla   v17.4s,  %[w1].4s,  v1.s[1]\n"  \
  "fmla   v18.4s,  %[w1].4s,  v1.s[3]\n"  \
  "fmla   v19.4s,  %[w1].4s,  v4.s[1]\n"  \
  "fmla   v20.4s,  %[w1].4s,  v4.s[3]\n"  \
  "fmla   v21.4s,  %[w1].4s,  v5.s[1]\n"  \
  "fmla   v22.4s,  %[w1].4s,  v5.s[3]\n"  \
  "ldp    q6, q7,   [%[r3]], #32      \n" \
  "fmla   v15.4s,  %[w7].4s,  v4.s[1]\n"  \
  "fmla   v16.4s,  %[w7].4s,  v4.s[3]\n"  \
  "fmla   v17.4s,  %[w7].4s,  v5.s[1]\n"  \
  "fmla   v18.4s,  %[w7].4s,  v5.s[3]\n"  \
  "ldr    d13,      [%[r3]]\n"            \
  "fmla   v15.4s,  %[w2].4s,  v0.s[2]\n"  \
  "fmla   v16.4s,  %[w2].4s,  v1.s[0]\n"  \
  "fmla   v17.4s,  %[w2].4s,  v1.s[2]\n"  \
  "fmla   v18.4s,  %[w2].4s,  v10.s[0]\n" \
  "fmla   v19.4s,  %[w2].4s,  v4.s[2]\n"  \
  "fmla   v20.4s,  %[w2].4s,  v5.s[0]\n"  \
  "fmla   v21.4s,  %[w2].4s,  v5.s[2]\n"  \
  "fmla   v22.4s,  %[w2].4s,  v12.s[0]\n" \
  "ldp    q8, q9,   [%[r4]], #32      \n" \
  "fmla   v15.4s,  %[w8].4s,  v4.s[2]\n"  \
  "fmla   v16.4s,  %[w8].4s,  v5.s[0]\n"  \
  "fmla   v17.4s,  %[w8].4s,  v5.s[2]\n"  \
  "fmla   v18.4s,  %[w8].4s,  v12.s[0]\n" \
  "ldr    d14,      [%[r4]]\n"            \
  "fmla   v15.4s,  %[w3].4s,  v2.s[0]\n"  \
  "fmla   v16.4s,  %[w3].4s,  v2.s[2]\n"  \
  "fmla   v17.4s,  %[w3].4s,  v3.s[0]\n"  \
  "fmla   v18.4s,  %[w3].4s,  v3.s[2]\n"  \
  "fmla   v19.4s,  %[w3].4s,  v6.s[0]\n"  \
  "fmla   v20.4s,  %[w3].4s,  v6.s[2]\n"  \
  "fmla   v21.4s,  %[w3].4s,  v7.s[0]\n"  \
  "fmla   v22.4s,  %[w3].4s,  v7.s[2]\n"  \
  "ldp    q0, q1,   [%[r0]], #32      \n" \
  "fmla   v15.4s,  %[w4].4s,  v2.s[1]\n"  \
  "fmla   v16.4s,  %[w4].4s,  v2.s[3]\n"  \
  "fmla   v17.4s,  %[w4].4s,  v3.s[1]\n"  \
  "fmla   v18.4s,  %[w4].4s,  v3.s[3]\n"  \
  "fmla   v19.4s,  %[w4].4s,  v6.s[1]\n"  \
  "fmla   v20.4s,  %[w4].4s,  v6.s[3]\n"  \
  "fmla   v21.4s,  %[w4].4s,  v7.s[1]\n"  \
  "fmla   v22.4s,  %[w4].4s,  v7.s[3]\n"  \
  "ldr    d10,      [%[r0]]\n"            \
  "fmla   v15.4s,  %[w5].4s,  v2.s[2]\n"  \
  "fmla   v16.4s,  %[w5].4s,  v3.s[0]\n"  \
  "fmla   v17.4s,  %[w5].4s,  v3.s[2]\n"  \
  "fmla   v18.4s,  %[w5].4s,  v11.s[0]\n" \
  "ldp    q4, q5,   [%[r2]], #32      \n" \
  "stp    q15, q16, [%[ptr_out0]], #32\n" \
  "fmla   v19.4s,  %[w5].4s,  v6.s[2]\n"  \
  "fmla   v20.4s,  %[w5].4s,  v7.s[0]\n"  \
  "fmla   v21.4s,  %[w5].4s,  v7.s[2]\n"  \
  "fmla   v22.4s,  %[w5].4s,  v13.s[0]\n" \
  "ldr    d12,      [%[r2]]\n"            \
  "stp    q17, q18, [%[ptr_out0]], #32\n" \
  "fmla   v19.4s,  %[w6].4s,  v8.s[0]\n"  \
  "fmla   v20.4s,  %[w6].4s,  v8.s[2]\n"  \
  "fmla   v21.4s,  %[w6].4s,  v9.s[0]\n"  \
  "fmla   v22.4s,  %[w6].4s,  v9.s[2]\n"

#define RESULT_FIRST                      \
  "fmla   v19.4s,  %[w7].4s,  v8.s[1]\n"  \
  "fmla   v20.4s,  %[w7].4s,  v8.s[3]\n"  \
  "fmla   v21.4s,  %[w7].4s,  v9.s[1]\n"  \
  "fmla   v22.4s,  %[w7].4s,  v9.s[3]\n"  \
  "fmla   v19.4s,  %[w8].4s,  v8.s[2]\n"  \
  "fmla   v20.4s,  %[w8].4s,  v9.s[0]\n"  \
  "fmla   v21.4s,  %[w8].4s,  v9.s[2]\n"  \
  "fmla   v22.4s,  %[w8].4s,  v14.s[0]\n" \
  "subs   %w[cnt], %w[cnt], #1\n"         \
  "stp    q19, q20, [%[ptr_out1]], #32\n" \
  "stp    q21, q22, [%[ptr_out1]], #32\n" \
  "bne    2b                          \n"

#define RESULT                            \
  "ldp    q15, q16, [%[ptr_out0]]     \n" \
  "fmla   v19.4s,  %[w7].4s,  v8.s[1]\n"  \
  "fmla   v20.4s,  %[w7].4s,  v8.s[3]\n"  \
  "fmla   v21.4s,  %[w7].4s,  v9.s[1]\n"  \
  "fmla   v22.4s,  %[w7].4s,  v9.s[3]\n"  \
  "ldp    q17, q18, [%[ptr_out0], #32]\n" \
  "fmla   v19.4s,  %[w8].4s,  v8.s[2]\n"  \
  "fmla   v20.4s,  %[w8].4s,  v9.s[0]\n"  \
  "fmla   v21.4s,  %[w8].4s,  v9.s[2]\n"  \
  "fmla   v22.4s,  %[w8].4s,  v14.s[0]\n" \
  "subs   %w[cnt], %w[cnt], #1\n"         \
  "stp    q19, q20, [%[ptr_out1]], #32\n" \
  "stp    q21, q22, [%[ptr_out1]], #32\n" \
  "bne    2b                          \n"

#define INIT_C1                                \
  "ldr    q21, [%[ptr_out0]]\n"                \
  "ld2  {v0.4s, v1.4s}, [%[r0]], #32\n"        \
  "ldr    d10,      [%[r0]]\n"                 \
  "ld2  {v4.4s, v5.4s}, [%[r2]], #32\n"        \
  "ldr    d12,      [%[r2]]\n"                 \
  "2:\n" /*  r0, r2, mul w0, get out r0, r1 */ \
  "ldr    q22, [%[ptr_out1]]\n"                \
  "fmla   v21.4s,  %[w0].4s,  v0.4s\n"         \
  "fmla   v22.4s,  %[w0].4s,  v4.4s\n"         \
  "ld2  {v2.4s, v3.4s}, [%[r1]], #32\n"

#define INIT_C1_FIRST                          \
  "ld2  {v0.4s, v1.4s}, [%[r0]], #32\n"        \
  "ldr    d10,      [%[r0]]\n"                 \
  "ld2  {v4.4s, v5.4s}, [%[r2]], #32\n"        \
  "ldr    d12,      [%[r2]]\n"                 \
  "2:\n" /*  r0, r2, mul w0, get out r0, r1 */ \
  "fmul   v21.4s,  %[w0].4s,  v0.4s\n"         \
  "fmul   v22.4s,  %[w0].4s,  v4.4s\n"         \
  "ld2  {v2.4s, v3.4s}, [%[r1]], #32\n"

#define COMPUTE_C1                                                         \
  /* r2 mul w6, get out r0*/                                               \
  "fmla   v21.4s,  %[w6].4s,  v4.4s\n"                                     \
  "ldr    d11,      [%[r1]]\n"                                             \
  "ext    v15.16b, v0.16b, v10.16b, #4\n"                                  \
  "ext    v16.16b, v4.16b, v12.16b, #4\n" /*  r0, r2, mul w1, get out r0*/ \
  "fmla   v21.4s,  %[w1].4s,  v1.4s\n"                                     \
  "fmla   v22.4s,  %[w1].4s,  v5.4s\n"                                     \
  "ld2  {v6.4s, v7.4s}, [%[r3]], #32\n" /*  r2 mul w7, get out r0 */       \
  "fmla   v21.4s,  %[w7].4s,  v5.4s\n"                                     \
  "ldr    d13,      [%[r3]]\n" /*  r0, r2, mul w2, get out r0, r1 */       \
  "fmla   v21.4s,  %[w2].4s,  v15.4s\n"                                    \
  "fmla   v22.4s,  %[w2].4s,  v16.4s\n"                                    \
  "ld2  {v8.4s, v9.4s}, [%[r4]], #32 \n" /*  r2, mul w8, get out r0 */     \
  "fmla   v21.4s,  %[w8].4s,  v16.4s\n"                                    \
  "ldr    d14,      [%[r4]]\n" /* r1, r3, mul w3, get out r0, r1 */        \
  "fmla   v21.4s,  %[w3].4s,  v2.4s\n"                                     \
  "fmla   v22.4s,  %[w3].4s,  v6.4s\n"                                     \
  "ext    v15.16b, v2.16b, v11.16b, #4\n"                                  \
  "ext    v16.16b, v6.16b, v13.16b, #4\n"                                  \
  "ld2  {v0.4s, v1.4s}, [%[r0]], #32\n"                                    \
  "fmla   v21.4s,  %[w4].4s,  v3.4s\n"                                     \
  "fmla   v22.4s,  %[w4].4s,  v7.4s\n"                                     \
  "ldr    d10,      [%[r0]]\n" /*  r1, r3, mul w5, get out r0, r1 */       \
  "fmla   v21.4s,  %[w5].4s,  v15.4s\n"                                    \
  "fmla   v22.4s,  %[w5].4s,  v16.4s\n"                                    \
  "ld2  {v4.4s, v5.4s}, [%[r2]], #32 \n"                                   \
  "ldr    d12,      [%[r2]]\n"                                             \
  "str    q21, [%[ptr_out0]], #16\n"

#define RESULT_C1                         \
  /*  r4, mul w6, get out r1 */           \
  "fmla   v22.4s,  %[w6].4s,  v8.4s  \n"  \
  "ext    v15.16b, v8.16b, v14.16b, #4\n" \
  "ldr    q21, [%[ptr_out0]]          \n" \
  "fmla   v22.4s,  %[w7].4s,  v9.4s  \n"  \
  "fmla   v22.4s,  %[w8].4s,  v15.4s \n"  \
  "subs   %w[cnt], %w[cnt], #1        \n" \
  "str    q22, [%[ptr_out1]], #16     \n" \
  "ldr    q22, [%[ptr_out1]]\n"           \
  "bne    2b                          \n"

#define RESULT_C1_FIRST                   \
  /*  r4, mul w6, get out r1 */           \
  "fmla   v22.4s,  %[w6].4s,  v8.4s  \n"  \
  "ext    v15.16b, v8.16b, v14.16b, #4\n" \
  "fmla   v22.4s,  %[w7].4s,  v9.4s  \n"  \
  "fmla   v22.4s,  %[w8].4s,  v15.4s \n"  \
  "subs   %w[cnt], %w[cnt], #1        \n" \
  "str    q22, [%[ptr_out1]], #16     \n" \
  "bne    2b                          \n"

#else
#define INIT_FIRST                                                             \
  "vld1.32    {d10-d13}, [%[wc0]]!       @ load w0, w1\n"                      \
  "vld1.32    {d14-d15}, [%[wc0]]!       @ load w2\n"                          \
  "vld1.32    {d0-d3}, [%[r0]]!          @ load r0\n"                          \
  "vld1.32    {d8},   [%[r0]]            @ load r0\n"   /* main loop */        \
  "0:                                    @ main loop\n" /* mul r0, with w0*/   \
  "vmul.f32   q8, q5, d0[0]              @ w0 * inr00\n"                       \
  "vmul.f32   q9, q5, d1[0]              @ w0 * inr02\n"                       \
  "vmul.f32   q10, q5, d2[0]             @ w0 * inr04\n"                       \
  "vmul.f32   q11, q5, d3[0]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  "vld1.32    {d4-d7}, [%[r2]]!          @ load r2\n"                          \
  "vmla.f32   q8, q6, d0[1]              @ w1 * inr01\n"                       \
  "vmla.f32   q9, q6, d1[1]              @ w1 * inr03\n"                       \
  "vmla.f32   q10, q6, d2[1]             @ w1 * inr05\n"                       \
  "vmla.f32   q11, q6, d3[1]             @ w1 * inr07\n"                       \
  "vld1.32    {d9},   [%[r2]]            @ load r2, 9th float\n"               \
  "vmla.f32   q8, q7, d1[0]              @ w2 * inr02\n"                       \
  "vmla.f32   q9, q7, d2[0]              @ w2 * inr04\n"                       \
  "vmla.f32   q10, q7, d3[0]             @ w2 * inr06\n"                       \
  "vmla.f32   q11, q7, d8[0]             @ w2 * inr08\n"                       \
  "sub    %[r2], %[r2], #32              @ r2 - 32\n" /* mul r2, with w0, w1*/ \
  "vld1.32    {d0-d3}, [%[r1]]!          @ load r1\n"                          \
  "vmul.f32   q12, q5, d4[0]             @ w0 * inr20\n"                       \
  "vmul.f32   q13, q5, d5[0]             @ w0 * inr22\n"                       \
  "vmul.f32   q14, q5, d6[0]             @ w0 * inr24\n"                       \
  "vmul.f32   q15, q5, d7[0]             @ w0 * inr26\n"

#define INIT                                                                   \
  "vld1.32    {d16-d19}, [%[ptr_out0]]!  @ load outr0\n"                       \
  "vld1.32    {d20-d23}, [%[ptr_out0]]   @ load outr0\n"                       \
  "vld1.32    {d10-d13}, [%[wc0]]!       @ load w0, w1\n"                      \
  "vld1.32    {d14-d15}, [%[wc0]]!       @ load w2\n"                          \
  "vld1.32    {d0-d3}, [%[r0]]!          @ load r0\n"                          \
  "vld1.32    {d8},   [%[r0]]            @ load r0\n" /* main loop */          \
  "sub    %[ptr_out0], %[ptr_out0], #32  @ ptr_out0 -32\n"                     \
  "0:                                    @ main loop\n" /* mul r0*/            \
  "vld1.32    {d24-d27}, [%[ptr_out1]]!  @ load outr1\n"                       \
  "vmla.f32   q8, q5, d0[0]              @ w0 * inr00\n"                       \
  "vld1.32    {d28-d31}, [%[ptr_out1]]   @ load outr1\n"                       \
  "vmla.f32   q9, q5, d1[0]              @ w0 * inr02\n"                       \
  "vmla.f32   q10, q5, d2[0]             @ w0 * inr04\n"                       \
  "vmla.f32   q11, q5, d3[0]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  "vld1.32    {d4-d7}, [%[r2]]!          @ load r2\n"                          \
  "vmla.f32   q8, q6, d0[1]              @ w1 * inr01\n"                       \
  "vmla.f32   q9, q6, d1[1]              @ w1 * inr03\n"                       \
  "vmla.f32   q10, q6, d2[1]             @ w1 * inr05\n"                       \
  "vmla.f32   q11, q6, d3[1]             @ w1 * inr07\n"                       \
  "vld1.32    {d9},   [%[r2]]            @ load r2, 9th float\n"               \
  "vmla.f32   q8, q7, d1[0]              @ w2 * inr02\n"                       \
  "vmla.f32   q9, q7, d2[0]              @ w2 * inr04\n"                       \
  "vmla.f32   q10, q7, d3[0]             @ w2 * inr06\n"                       \
  "vmla.f32   q11, q7, d8[0]             @ w2 * inr08\n"                       \
  "sub    %[r2], %[r2], #32              @ r2 - 32\n" /* mul r2, with w0, w1*/ \
  "vld1.32    {d0-d3}, [%[r1]]!          @ load r1\n"                          \
  "vmla.f32   q12, q5, d4[0]             @ w0 * inr20\n"                       \
  "vmla.f32   q13, q5, d5[0]             @ w0 * inr22\n"                       \
  "vmla.f32   q14, q5, d6[0]             @ w0 * inr24\n"                       \
  "vmla.f32   q15, q5, d7[0]             @ w0 * inr26\n"

#define COMPUTE                                                                \
  "vld1.32    {d8},   [%[r1]]            @ load r1, 9th float\n"               \
  "vmla.f32   q12, q6, d4[1]             @ w1 * inr21\n"                       \
  "vmla.f32   q13, q6, d5[1]             @ w1 * inr23\n"                       \
  "vmla.f32   q14, q6, d6[1]             @ w1 * inr25\n"                       \
  "vmla.f32   q15, q6, d7[1]             @ w1 * inr27\n"                       \
  "vld1.32    {d10-d13}, [%[wc0]]!       @ load w3, w4, to q5, q6\n"           \
  "vmla.f32   q12, q7, d5[0]             @ w2 * inr22\n"                       \
  "vmla.f32   q13, q7, d6[0]             @ w2 * inr24\n"                       \
  "vmla.f32   q14, q7, d7[0]             @ w2 * inr26\n"                       \
  "vmla.f32   q15, q7, d9[0]             @ w2 * inr28\n"                       \
  "vld1.32    {d14-d15}, [%[wc0]]!       @ load w5, to q7\n" /* mul r1, with*/ \
  "vmla.f32   q8, q5, d0[0]              @ w3 * inr10\n"                       \
  "vmla.f32   q9, q5, d1[0]              @ w3 * inr12\n"                       \
  "vmla.f32   q10, q5, d2[0]             @ w3 * inr14\n"                       \
  "vmla.f32   q11, q5, d3[0]             @ w3 * inr16\n"                       \
  "vld1.32    {d4-d7}, [%[r3]]!          @ load r3, 8 float\n"                 \
  "vmla.f32   q8, q6, d0[1]              @ w4 * inr11\n"                       \
  "vmla.f32   q9, q6, d1[1]              @ w4 * inr13\n"                       \
  "vmla.f32   q10, q6, d2[1]             @ w4 * inr15\n"                       \
  "vmla.f32   q11, q6, d3[1]             @ w4 * inr17\n"                       \
  "vld1.32    {d9},   [%[r3]]            @ load r3, 9th float\n"               \
  "vmla.f32   q8, q7, d1[0]              @ w5 * inr12\n"                       \
  "vmla.f32   q9, q7, d2[0]              @ w5 * inr14\n"                       \
  "vmla.f32   q10, q7, d3[0]             @ w5 * inr16\n"                       \
  "vmla.f32   q11, q7, d8[0]             @ w5 * inr18\n"

#define RESULT_FIRST                                                           \
  /* mul r3, with w3, w4, w5 */                                                \
  "vld1.32    {d0-d3}, [%[r2]]!          @ load r2\n"                          \
  "vmla.f32   q12, q5, d4[0]             @ w3 * inr30\n"                       \
  "vmla.f32   q13, q5, d5[0]             @ w3 * inr32\n"                       \
  "vmla.f32   q14, q5, d6[0]             @ w3 * inr34\n"                       \
  "vmla.f32   q15, q5, d7[0]             @ w3 * inr36\n"                       \
  "vld1.32    {d8},   [%[r2]]            @ load r2, 9th float\n"               \
  "vmla.f32   q12, q6, d4[1]             @ w4 * inr31\n"                       \
  "vmla.f32   q13, q6, d5[1]             @ w4 * inr33\n"                       \
  "vmla.f32   q14, q6, d6[1]             @ w4 * inr35\n"                       \
  "vmla.f32   q15, q6, d7[1]             @ w4 * inr37\n"                       \
  "vld1.32    {d10-d13}, [%[wc0]]!       @ load w6, w7\n"                      \
  "vmla.f32   q12, q7, d5[0]             @ w5 * inr32\n"                       \
  "vmla.f32   q13, q7, d6[0]             @ w5 * inr34\n"                       \
  "vmla.f32   q14, q7, d7[0]             @ w5 * inr36\n"                       \
  "vmla.f32   q15, q7, d9[0]             @ w5 * inr38\n"                       \
  "vld1.32    {d14-d15}, [%[wc0]]!       @ load w8\n" /* mul r2, with w6, w7*/ \
  "vmla.f32   q8, q5, d0[0]              @ w6 * inr20\n"                       \
  "vmla.f32   q9, q5, d1[0]              @ w6 * inr22\n"                       \
  "vmla.f32   q10, q5, d2[0]             @ w6 * inr24\n"                       \
  "vmla.f32   q11, q5, d3[0]             @ w6 * inr26\n"                       \
  "vld1.32    {d4-d7}, [%[r4]]!          @ load r4\n"                          \
  "vmla.f32   q8, q6, d0[1]              @ w7 * inr21\n"                       \
  "vmla.f32   q9, q6, d1[1]              @ w7 * inr23\n"                       \
  "vmla.f32   q10, q6, d2[1]             @ w7 * inr25\n"                       \
  "vmla.f32   q11, q6, d3[1]             @ w7 * inr27\n"                       \
  "vld1.32    {d9},   [%[r4]]            @ load r4, 9th float\n"               \
  "vmla.f32   q8, q7, d1[0]              @ w8 * inr22\n"                       \
  "vmla.f32   q9, q7, d2[0]              @ w8 * inr24\n"                       \
  "vmla.f32   q10, q7, d3[0]             @ w8 * inr26\n"                       \
  "vmla.f32   q11, q7, d8[0]             @ w8 * inr28\n"                       \
  "sub    %[wc0], %[wc0], #144           @ wc0 - 144\n" /* mul r4, with w6*/   \
  "vld1.32    {d0-d3}, [%[r0]]!          @ load r0\n"                          \
  "vmla.f32   q12, q5, d4[0]             @ w3 * inr40\n"                       \
  "vst1.32    {d16-d19}, [%[ptr_out0]]!  @ save r00, r01\n"                    \
  "vmla.f32   q13, q5, d5[0]             @ w3 * inr42\n"                       \
  "vst1.32    {d20-d23}, [%[ptr_out0]]!  @ save r02, r03\n"                    \
  "vmla.f32   q14, q5, d6[0]             @ w3 * inr44\n"                       \
  "vmla.f32   q15, q5, d7[0]             @ w3 * inr46\n"                       \
  "vld1.32    {d8},   [%[r0]]            @ load r0, 9th float\n"               \
  "vmla.f32   q12, q6, d4[1]             @ w4 * inr41\n"                       \
  "vmla.f32   q13, q6, d5[1]             @ w4 * inr43\n"                       \
  "vmla.f32   q14, q6, d6[1]             @ w4 * inr45\n"                       \
  "vmla.f32   q15, q6, d7[1]             @ w4 * inr47\n"                       \
  "vld1.32    {d10-d13}, [%[wc0]]!       @ load w0, w1\n"                      \
  "vmla.f32   q12, q7, d5[0]             @ w5 * inr42\n"                       \
  "vmla.f32   q13, q7, d6[0]             @ w5 * inr44\n"                       \
  "vmla.f32   q14, q7, d7[0]             @ w5 * inr46\n"                       \
  "vmla.f32   q15, q7, d9[0]             @ w5 * inr48\n"                       \
  "subs   %[cnt], #1                     @ loop count--\n"                     \
  "vld1.32    {d14-d15}, [%[wc0]]!       @ load w2\n"                          \
  "vst1.32    {d24-d27}, [%[ptr_out1]]!  @ save r10, r11\n"                    \
  "vst1.32    {d28-d31}, [%[ptr_out1]]!  @ save r12, r13\n"                    \
  "bne    0b                             @ jump to main loop\n"

#define RESULT                                                                 \
  "sub    %[ptr_out1], %[ptr_out1], #32  @ ptr_out1 - 32\n" /* mul r3, with */ \
  "vld1.32    {d0-d3}, [%[r2]]!          @ load r2\n"                          \
  "vmla.f32   q12, q5, d4[0]             @ w3 * inr30\n"                       \
  "vmla.f32   q13, q5, d5[0]             @ w3 * inr32\n"                       \
  "vmla.f32   q14, q5, d6[0]             @ w3 * inr34\n"                       \
  "vmla.f32   q15, q5, d7[0]             @ w3 * inr36\n"                       \
  "vld1.32    {d8},   [%[r2]]            @ load r2, 9th float\n"               \
  "vmla.f32   q12, q6, d4[1]             @ w4 * inr31\n"                       \
  "vmla.f32   q13, q6, d5[1]             @ w4 * inr33\n"                       \
  "vmla.f32   q14, q6, d6[1]             @ w4 * inr35\n"                       \
  "vmla.f32   q15, q6, d7[1]             @ w4 * inr37\n"                       \
  "vld1.32    {d10-d13}, [%[wc0]]!       @ load w6, w7\n"                      \
  "vmla.f32   q12, q7, d5[0]             @ w5 * inr32\n"                       \
  "vmla.f32   q13, q7, d6[0]             @ w5 * inr34\n"                       \
  "vmla.f32   q14, q7, d7[0]             @ w5 * inr36\n"                       \
  "vmla.f32   q15, q7, d9[0]             @ w5 * inr38\n"                       \
  "vld1.32    {d14-d15}, [%[wc0]]!       @ load w8\n" /* mul r2, with w6, w7*/ \
  "vmla.f32   q8, q5, d0[0]              @ w6 * inr20\n"                       \
  "vmla.f32   q9, q5, d1[0]              @ w6 * inr22\n"                       \
  "vmla.f32   q10, q5, d2[0]             @ w6 * inr24\n"                       \
  "vmla.f32   q11, q5, d3[0]             @ w6 * inr26\n"                       \
  "vld1.32    {d4-d7}, [%[r4]]!          @ load r4\n"                          \
  "vmla.f32   q8, q6, d0[1]              @ w7 * inr21\n"                       \
  "vmla.f32   q9, q6, d1[1]              @ w7 * inr23\n"                       \
  "vmla.f32   q10, q6, d2[1]             @ w7 * inr25\n"                       \
  "vmla.f32   q11, q6, d3[1]             @ w7 * inr27\n"                       \
  "vld1.32    {d9},   [%[r4]]            @ load r4, 9th float\n"               \
  "vmla.f32   q8, q7, d1[0]              @ w8 * inr22\n"                       \
  "vmla.f32   q9, q7, d2[0]              @ w8 * inr24\n"                       \
  "vmla.f32   q10, q7, d3[0]             @ w8 * inr26\n"                       \
  "vmla.f32   q11, q7, d8[0]             @ w8 * inr28\n"                       \
  "sub    %[wc0], %[wc0], #144           @ wc0 - 144\n" /* mul r4, with w6*/   \
  "vld1.32    {d0-d3}, [%[r0]]!          @ load r0\n"                          \
  "vmla.f32   q12, q5, d4[0]             @ w3 * inr40\n"                       \
  "vst1.32    {d16-d19}, [%[ptr_out0]]!  @ save r00, r01\n"                    \
  "vmla.f32   q13, q5, d5[0]             @ w3 * inr42\n"                       \
  "vst1.32    {d20-d23}, [%[ptr_out0]]!  @ save r02, r03\n"                    \
  "vmla.f32   q14, q5, d6[0]             @ w3 * inr44\n"                       \
  "vmla.f32   q15, q5, d7[0]             @ w3 * inr46\n"                       \
  "vld1.32    {d8},   [%[r0]]            @ load r0, 9th float\n"               \
  "vmla.f32   q12, q6, d4[1]             @ w4 * inr41\n"                       \
  "vmla.f32   q13, q6, d5[1]             @ w4 * inr43\n"                       \
  "vmla.f32   q14, q6, d6[1]             @ w4 * inr45\n"                       \
  "vmla.f32   q15, q6, d7[1]             @ w4 * inr47\n"                       \
  "vld1.32    {d10-d13}, [%[wc0]]!       @ load w0, w1\n"                      \
  "vmla.f32   q12, q7, d5[0]             @ w5 * inr42\n"                       \
  "vmla.f32   q13, q7, d6[0]             @ w5 * inr44\n"                       \
  "vmla.f32   q14, q7, d7[0]             @ w5 * inr46\n"                       \
  "vmla.f32   q15, q7, d9[0]             @ w5 * inr48\n"                       \
  "vld1.32    {d14-d15}, [%[wc0]]!       @ load w2\n"                          \
  "vst1.32    {d24-d27}, [%[ptr_out1]]!  @ save r10, r11\n"                    \
  "vst1.32    {d28-d31}, [%[ptr_out1]]!  @ save r12, r13\n"                    \
  "vld1.32    {d16-d19}, [%[ptr_out0]]!  @ load outr0\n"                       \
  "vld1.32    {d20-d23}, [%[ptr_out0]]   @ load outr0\n"                       \
  "sub    %[ptr_out0], %[ptr_out0], #32  @ ptr_out0 - 32\n"                    \
  "subs   %[cnt], #1                     @ loop count--\n"                     \
  "bne    0b                             @ jump to main loop\n"

#define INIT_C1                                                        \
  "0:                                @ main loop\n"                    \
  "vld1.32 {d24-d27}, [%[ptr_out0]]  @ load or00, or01\n"              \
  "vld1.32 {d28-d31}, [%[ptr_out1]]  @ load or10, or11\n"              \
  "vld2.32    {d6-d9},    [%[r2]]!   @ load r2\n"                      \
  "vld2.32    {d10-d13},  [%[r2]]!   @ load r2\n"                      \
  "vld1.32    {d22},  [%[r2]]        @ load 16th float\n" /* r2 * w2*/ \
  "vmla.f32   q12,    q4, %e[w2][1]  @ w21 * r2\n"                     \
  "vmla.f32   q13,    q6, %e[w2][1]  @ w21 * r2\n"                     \
  "vld2.32    {d14-d17},    [%[r0]]! @ load r0\n"                      \
  "vmla.f32   q14,    q4, %e[w0][1]  @ w01 * r2\n"                     \
  "vmla.f32   q15,    q6, %e[w0][1]  @ w01 * r2\n"

#define INIT_C1_FIRST                                     \
  "0:                                @ main loop\n"       \
  "vld2.32    {d6-d9},    [%[r2]]!   @ load r2\n"         \
  "vld2.32    {d10-d13},  [%[r2]]!   @ load r2\n"         \
  "vld1.32    {d22},  [%[r2]]        @ load 16th float\n" \
  "vmul.f32   q12,    q4, %e[w2][1]  @ w21 * r2\n"        \
  "vmul.f32   q13,    q6, %e[w2][1]  @ w21 * r2\n"        \
  "vld2.32    {d14-d17},    [%[r0]]! @ load r0\n"         \
  "vmul.f32   q14,    q4, %e[w0][1]  @ w01 * r2\n"        \
  "vmul.f32   q15,    q6, %e[w0][1]  @ w01 * r2\n"

#define COMPUTE_C1                                                           \
  "vext.32    q4, q3, q5, #1         @ r2, shift left 1\n"                   \
  "vext.32    q6, q5, q11, #1        @ r2, shift left 1\n"                   \
  "vmla.f32   q12,    q3, %e[w2][0]  @ w20 * r2\n"                           \
  "vmla.f32   q13,    q5, %e[w2][0]  @ w20 * r2\n"                           \
  "vld2.32    {d18-d21},  [%[r0]]!   @ load r0\n"                            \
  "vmla.f32   q14,    q3, %e[w0][0]  @ w00 * r2\n"                           \
  "vmla.f32   q15,    q5, %e[w0][0]  @ w00 * r2\n"                           \
  "vld1.32    {d22},  [%[r0]]        @ load 16th float\n"                    \
  "vmla.f32   q12,    q4, %f[w2][0]  @ w22 * r2\n"                           \
  "vmla.f32   q14,    q4, %f[w0][0]  @ w02 * r2\n"                           \
  "vld2.32    {d6-d9},    [%[r3]]!   @ load r3\n"                            \
  "vmla.f32   q13,    q6, %f[w2][0]  @ w22 * r2\n"                           \
  "vmla.f32   q15,    q6, %f[w0][0]  @ w02 * r2\n"                           \
  "vld2.32    {d10-d13},  [%[r3]]!   @ load r3\n" /* r0 * w0, get or0, r3 */ \
  "vmla.f32   q12,    q8, %e[w0][1]      @ w01 * r0\n"                       \
  "vmla.f32   q13,    q10, %e[w0][1]     @ w01 * r0\n"                       \
  "vext.32    q8, q7, q9, #1             @ r0, shift left 1\n"               \
  "vext.32    q10, q9, q11, #1           @ r0, shift left 1\n"               \
  "vld1.32    {d22},  [%[r3]]            @ load 16th float\n"                \
  "vmla.f32   q14,    q4, %e[w1][1]      @ w11 * r3\n"                       \
  "vmla.f32   q15,    q6, %e[w1][1]      @ w11 * r3\n"                       \
  "vmla.f32   q12,    q7, %e[w0][0]      @ w00 * r0\n"                       \
  "vmla.f32   q13,    q9, %e[w0][0]      @ w00 * r0\n"                       \
  "vext.32    q4, q3, q5, #1             @ r3, shift left 1\n"               \
  "vext.32    q6, q5, q11, #1            @ r3, shift left 1\n"               \
  "vmla.f32   q14,    q3, %e[w1][0]      @ w10 * r3\n"                       \
  "vmla.f32   q15,    q5, %e[w1][0]      @ w10 * r3\n"                       \
  "vmla.f32   q12,    q8, %f[w0][0]      @ w02 * r0\n"                       \
  "vld2.32    {d14-d17},  [%[r1]]!       @ load r1\n"                        \
  "vmla.f32   q13,    q10,%f[w0][0]      @ w02 * r0\n"                       \
  "vld2.32    {d18-d21},  [%[r1]]!       @ load r1\n"                        \
  "vmla.f32   q14,    q4, %f[w1][0]      @ w12 * r3\n"                       \
  "vld2.32    {d6-d9},    [%[r4]]!       @ load r4\n"                        \
  "vmla.f32   q15,    q6, %f[w1][0]      @ w12 * r3\n"                       \
  "vld2.32    {d10-d13},  [%[r4]]!       @ load r4\n"                        \
  "vld1.32    {d22},  [%[r1]]            @ load 16th float\n" /* r1 * w1  */ \
  "vmla.f32   q12,    q8, %e[w1][1]      @ w11 * r1\n"                       \
  "vmla.f32   q13,    q10, %e[w1][1]     @ w11 * r1\n"                       \
  "vext.32    q8, q7, q9, #1             @ r1, shift left 1\n"               \
  "vext.32    q10, q9, q11, #1           @ r1, shift left 1\n"               \
  "vmla.f32   q14,    q4, %e[w2][1]      @ w21 * r4\n"                       \
  "vmla.f32   q15,    q6, %e[w2][1]      @ w21 * r4\n"                       \
  "vld1.32    {d22},  [%[r4]]            @ load 16th float\n"                \
  "vmla.f32   q12,    q7, %e[w1][0]      @ w10 * r1\n"                       \
  "vmla.f32   q13,    q9, %e[w1][0]      @ w10 * r1\n"                       \
  "vext.32    q4, q3, q5, #1             @ r1, shift left 1\n"               \
  "vext.32    q6, q5, q11, #1            @ r1, shift left 1\n"               \
  "vmla.f32   q14,    q3, %e[w2][0]      @ w20 * r4\n"                       \
  "vmla.f32   q15,    q5, %e[w2][0]      @ w20 * r4\n"                       \
  "vmla.f32   q12,    q8, %f[w1][0]      @ w12 * r1\n"                       \
  "vmla.f32   q13,    q10, %f[w1][0]     @ w12 * r1\n"                       \
  "vmla.f32   q14,    q4, %f[w2][0]      @ w22 * r4\n"                       \
  "vmla.f32   q15,    q6, %f[w2][0]      @ w22 * r4\n"                       \
  "vst1.32    {d24-d27},  [%[ptr_out0]]! @ save or0\n"                       \
  "vst1.32    {d28-d31},  [%[ptr_out1]]! @ save or0\n"                       \
  "subs   %[cnt], #1                     @ loop count -1\n"                  \
  "bne    0b                             @ jump to main loop\n"
#endif
void conv_3x3s2_direct_fp32_c3(const float* i_data,
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
                               ARMContext* ctx);
#ifdef __aarch64__
void conv_3x3s2_direct_fp32_c3_a53(const float* i_data,
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
                                   ARMContext* ctx);
#endif
void conv_3x3s2_direct_fp32(const float* i_data,
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
                            ARMContext* ctx) {
  //! 3x3s2 convolution, implemented by direct algorithm
  //! prepack input to tmp buffer
  //! write output to tmp buffer
  if (ic == 3 && (oc % 4 == 0)) {
#ifdef __aarch64__
    if (ctx->arch() == kA53 || ctx->arch() == kA35) {
      conv_3x3s2_direct_fp32_c3_a53(i_data,
                                    o_data,
                                    bs,
                                    oc,
                                    oh,
                                    ow,
                                    ic,
                                    ih,
                                    win,
                                    weights,
                                    bias,
                                    param,
                                    ctx);
    } else {
      conv_3x3s2_direct_fp32_c3(i_data,
                                o_data,
                                bs,
                                oc,
                                oh,
                                ow,
                                ic,
                                ih,
                                win,
                                weights,
                                bias,
                                param,
                                ctx);
    }
#else
    conv_3x3s2_direct_fp32_c3(
        i_data, o_data, bs, oc, oh, ow, ic, ih, win, weights, bias, param, ctx);
#endif
    return;
  }
  auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  const int threads = ctx->threads();
  int l2_size = ctx->llc_size() / sizeof(float);
  const int pad_w = paddings[2];
  const int pad_h = paddings[0];
  const int wout_round = ROUNDUP(ow, OUT_W_BLOCK);
  const int win_round = wout_round * 2 /*stride_w*/ + 1;
  bool flag_bias = param.bias != nullptr;

  //! get h block
  //! win_round * ic * hin_r_block + wout_round * OUT_C_BLOCK * hout_r_block
  //! * threads = l2_size
  //! win_round = 2 * wout_round + 1
  //! hin_r_block = 2 * hout_r_block + 1
  int hout_r_block =
      (l2_size - 2 * wout_round * ic - ic) /
      ((4 * wout_round + 2) * ic + wout_round * OUT_C_BLOCK * threads);
  hout_r_block = hout_r_block > oh ? oh : hout_r_block;
  hout_r_block = (hout_r_block / OUT_H_BLOCK) * OUT_H_BLOCK;
  hout_r_block = hout_r_block < OUT_H_BLOCK ? OUT_H_BLOCK : hout_r_block;

  const int hin_r_block = hout_r_block * 2 /*stride_h*/ + 1;

  int in_len = win_round * ic;
  int pre_in_size = hin_r_block * in_len;
  int pre_out_size = OUT_C_BLOCK * hout_r_block * wout_round;

  float* tmp_work_space = ctx->workspace_data<float>();
  float ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(float) * win_round);

  //! l2_cache start
  float* pre_din = tmp_work_space;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;
  int w_stride = ic * 9; /*kernel_w * kernel_h*/
  int w_stride_chin = OUT_C_BLOCK * 9;

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int c_remain = oc - (oc / OUT_C_BLOCK) * OUT_C_BLOCK;
  int c_round_down = (oc / OUT_C_BLOCK) * OUT_C_BLOCK;

  int out_row_stride = OUT_C_BLOCK * wout_round;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }

  for (int n = 0; n < bs; ++n) {
    const float* din_batch = i_data + n * ic * size_in_channel;
    float* dout_batch = o_data + n * oc * size_out_channel;
    for (int h = 0; h < oh; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > oh) {
        h_kernel = oh - h;
      }

      int hs = h * 2 /*stride_h*/ - pad_h;
      int he = hs + h_kernel * 2 /*stride_h*/ + 1;

      prepack_input_nxw(
          din_batch, pre_din, 0, ic, hs, he, ws, we, ic, win, ih, ptr_zero);

      const float* cblock_inr0 = pre_din;
      const float* cblock_inr1 = cblock_inr0 + in_len;
      const float* cblock_inr2 = cblock_inr1 + in_len;
      const float* cblock_inr3 = cblock_inr2 + in_len;
      const float* cblock_inr4 = cblock_inr3 + in_len;

      LITE_PARALLEL_COMMON_BEGIN(c, tid, c_round_down, 0, OUT_C_BLOCK) {
#ifdef LITE_USE_THREAD_POOL
        float* pre_out = pre_din + pre_in_size + tid * pre_out_size;
#elif defined(ARM_WITH_OMP)
        float* pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float* pre_out = pre_din + pre_in_size;
#endif
        const float* block_inr0 = cblock_inr0;
        const float* block_inr1 = cblock_inr1;
        const float* block_inr2 = cblock_inr2;
        const float* block_inr3 = cblock_inr3;
        const float* block_inr4 = cblock_inr4;

        const float* weight_c = weights + c * w_stride;
        const float* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = bias + c;
        }

        for (int hk = 0; hk < h_kernel; hk += OUT_H_BLOCK) {
          const float* wc0 = weight_c;

          const float* inr0 = block_inr0;
          const float* inr1 = block_inr1;
          const float* inr2 = block_inr2;
          const float* inr3 = block_inr3;
          const float* inr4 = block_inr4;

          float* pre_out0 = pre_out + hk * out_row_stride;
          float* pre_out1 = pre_out0 + out_row_stride;
#ifdef __aarch64__
          // first
          float* ptr_out0 = pre_out0;
          float* ptr_out1 = pre_out1;
          float32x4_t w0 = vld1q_f32(wc0);       // w0, v23
          float32x4_t w1 = vld1q_f32(wc0 + 4);   // w1, v24
          float32x4_t w2 = vld1q_f32(wc0 + 8);   // w2, v25
          float32x4_t w3 = vld1q_f32(wc0 + 12);  // w3, v26
          float32x4_t w4 = vld1q_f32(wc0 + 16);  // w4, v27
          float32x4_t w5 = vld1q_f32(wc0 + 20);  // w5, v28
          float32x4_t w6 = vld1q_f32(wc0 + 24);  // w6, v29
          float32x4_t w7 = vld1q_f32(wc0 + 28);  // w7, v30
          float32x4_t w8 = vld1q_f32(wc0 + 32);  // w8, v31
          const float* r0 = inr0;
          const float* r1 = inr1;
          const float* r2 = inr2;
          const float* r3 = inr3;
          const float* r4 = inr4;

          int cnt = w_loop;
          // clang-format off
          asm volatile(
            INIT_FIRST COMPUTE RESULT_FIRST
            : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1),
              [r2] "+r"(r2),[r3] "+r"(r3), [r4] "+r"(r4),
              [ptr_out0] "+r"(ptr_out0),
              [ptr_out1] "+r"(ptr_out1)
            : [w0] "w"(w0),
              [w1] "w"(w1), [w2] "w"(w2),
              [w3] "w"(w3), [w4] "w"(w4),
              [w5] "w"(w5), [w6] "w"(w6),
              [w7] "w"(w7), [w8] "w"(w8)
            : "cc","memory","v0","v1","v2","v3","v4",
              "v5","v6","v7","v8","v9","v10","v11","v12","v13",
              "v14","v15","v16","v17","v18","v19","v20","v21","v22");
          // clang-format on
          wc0 += 9 * OUT_C_BLOCK;
          inr0 += win_round;
          inr1 += win_round;
          inr2 += win_round;
          inr3 += win_round;
          inr4 += win_round;

          for (int i = 0; i < ic - 1; ++i) {
            ptr_out0 = pre_out0;
            ptr_out1 = pre_out1;

            w0 = vld1q_f32(wc0);       // w0, v23
            w1 = vld1q_f32(wc0 + 4);   // w1, v24
            w2 = vld1q_f32(wc0 + 8);   // w2, v25
            w3 = vld1q_f32(wc0 + 12);  // w3, v26
            w4 = vld1q_f32(wc0 + 16);  // w4, v27
            w5 = vld1q_f32(wc0 + 20);  // w5, v28
            w6 = vld1q_f32(wc0 + 24);  // w6, v29
            w7 = vld1q_f32(wc0 + 28);  // w7, v30
            w8 = vld1q_f32(wc0 + 32);  // w8, v31

            r0 = inr0;
            r1 = inr1;
            r2 = inr2;
            r3 = inr3;
            r4 = inr4;

            int cnt = w_loop;
            // clang-format off
            asm volatile(
            INIT COMPUTE RESULT
            : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1),
              [r2] "+r"(r2),[r3] "+r"(r3), [r4] "+r"(r4),
              [ptr_out0] "+r"(ptr_out0),
              [ptr_out1] "+r"(ptr_out1)
            : [w0] "w"(w0),
              [w1] "w"(w1), [w2] "w"(w2),
              [w3] "w"(w3), [w4] "w"(w4),
              [w5] "w"(w5), [w6] "w"(w6),
              [w7] "w"(w7), [w8] "w"(w8)
            : "cc","memory","v0","v1","v2","v3","v4",
              "v5","v6","v7","v8","v9","v10","v11","v12","v13",
              "v14","v15","v16","v17","v18","v19","v20","v21","v22");
            // clang-format on
            wc0 += 9 * OUT_C_BLOCK;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
#else   // not __aarch64__
          const float* wc00 = wc0;
          float* ptr_out0 = pre_out0;
          float* ptr_out1 = pre_out1;
          const float* r0 = inr0;
          const float* r1 = inr1;
          const float* r2 = inr2;
          const float* r3 = inr3;
          const float* r4 = inr4;
          int cnt = w_loop;
          // clang-format off
          asm volatile(
            INIT_FIRST COMPUTE RESULT_FIRST
          : [cnt] "+r"(cnt),
              [r0] "+r"(r0),[r1] "+r"(r1),
              [r2] "+r"(r2),[r3] "+r"(r3),
              [r4] "+r"(r4),
              [ptr_out0] "+r"(ptr_out0),
              [ptr_out1] "+r"(ptr_out1),
              [wc0] "+r"(wc00)
          :
          : "cc","memory","q0","q1","q2","q3","q4",
              "q5","q6","q7","q8","q9","q10",
              "q11","q12","q13","q14","q15"
          );
          // clang-format on
          wc0 += w_stride_chin;
          inr0 += win_round;
          inr1 += win_round;
          inr2 += win_round;
          inr3 += win_round;
          inr4 += win_round;

          for (int i = 0; i < ic - 1; ++i) {
            wc00 = wc0;
            ptr_out0 = pre_out0;
            ptr_out1 = pre_out1;

            r0 = inr0;
            r1 = inr1;
            r2 = inr2;
            r3 = inr3;
            r4 = inr4;

            cnt = w_loop;
            // clang-format off
            asm volatile(
              INIT COMPUTE RESULT
            : [cnt] "+r"(cnt),
              [r0] "+r"(r0),[r1] "+r"(r1),
              [r2] "+r"(r2),[r3] "+r"(r3),
              [r4] "+r"(r4),
              [ptr_out0] "+r"(ptr_out0),
              [ptr_out1] "+r"(ptr_out1),
              [wc0] "+r"(wc00)
            :
            : "cc","memory","q0","q1","q2","q3","q4",
              "q5","q6","q7","q8","q9","q10",
              "q11","q12","q13","q14","q15"
            );
            // clang-format on
            wc0 += w_stride_chin;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
#endif  // __aarch64__
          block_inr0 = block_inr4;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }
        write_to_output_c4_fp32(pre_out,
                                dout_batch,
                                c,
                                c + OUT_C_BLOCK,
                                h,
                                h + h_kernel,
                                0,
                                wout_round,
                                oc,
                                oh,
                                ow,
                                false,
                                nullptr,
                                &act_param,
                                bias_ptr);
      }
      LITE_PARALLEL_COMMON_END();

      LITE_PARALLEL_BEGIN(c, tid, c_remain) {
#ifdef LITE_USE_THREAD_POOL
        float* pre_out = pre_din + pre_in_size + tid * pre_out_size;
#elif defined(ARM_WITH_OMP)
        float* pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float* pre_out = pre_din + pre_in_size;
#endif

        const float* block_inr0 = cblock_inr0;
        const float* block_inr1 = cblock_inr1;
        const float* block_inr2 = cblock_inr2;
        const float* block_inr3 = cblock_inr3;
        const float* block_inr4 = cblock_inr4;

        //! get weights ptr of remained
        const float* weight_c = weights + c_round_down * w_stride;

        //! fill bias to one channel
        const float* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = bias + c_round_down + c;
        }

        for (int hk = 0; hk < h_kernel; hk += OUT_H_BLOCK) {
          const float* wc0 = weight_c;

          const float* inr0 = block_inr0;
          const float* inr1 = block_inr1;
          const float* inr2 = block_inr2;
          const float* inr3 = block_inr3;
          const float* inr4 = block_inr4;

          float* pre_out0 = pre_out + hk * wout_round;
          float* pre_out1 = pre_out0 + wout_round;
#ifdef __aarch64__
          float* ptr_out0 = pre_out0;
          float* ptr_out1 = pre_out1;

          //! get valid weights of current output channel
          float32x4_t w0 = vdupq_n_f32(wc0[c]);       // w0, v23
          float32x4_t w1 = vdupq_n_f32(wc0[c + 4]);   // w1, v24
          float32x4_t w2 = vdupq_n_f32(wc0[c + 8]);   // w2, v25
          float32x4_t w3 = vdupq_n_f32(wc0[c + 12]);  // w3, v26
          float32x4_t w4 = vdupq_n_f32(wc0[c + 16]);  // w4, v27
          float32x4_t w5 = vdupq_n_f32(wc0[c + 20]);  // w5, v28
          float32x4_t w6 = vdupq_n_f32(wc0[c + 24]);  // w6, v29
          float32x4_t w7 = vdupq_n_f32(wc0[c + 28]);  // w7, v30
          float32x4_t w8 = vdupq_n_f32(wc0[c + 32]);  // w8, v31

          const float* r0 = inr0;
          const float* r1 = inr1;
          const float* r2 = inr2;
          const float* r3 = inr3;
          const float* r4 = inr4;

          int cnt = w_loop;
          // clang-format off
          asm volatile(
              INIT_C1_FIRST COMPUTE_C1 RESULT_C1_FIRST
          : [cnt] "+r"(cnt),
            [r0] "+r"(r0),[r1] "+r"(r1),
            [r2] "+r"(r2),[r3] "+r"(r3),
            [r4] "+r"(r4),
            [ptr_out0] "+r"(ptr_out0),
            [ptr_out1] "+r"(ptr_out1)
          : [w0] "w"(w0),[w1] "w"(w1),[w2] "w"(w2),
            [w3] "w"(w3),[w4] "w"(w4),[w5] "w"(w5),
            [w6] "w"(w6),[w7] "w"(w7),[w8] "w"(w8)
            : "cc","memory","v0","v1","v2","v3",
              "v4","v5","v6","v7","v8","v9","v10","v11",
              "v12","v13","v14","v15","v16","v21","v22");
          // clang-format on
          wc0 += 36;
          inr0 += win_round;
          inr1 += win_round;
          inr2 += win_round;
          inr3 += win_round;
          inr4 += win_round;
          for (int i = 0; i < ic - 1; ++i) {
            ptr_out0 = pre_out0;
            ptr_out1 = pre_out1;

            //! get valid weights of current output channel
            w0 = vdupq_n_f32(wc0[c]);       // w0, v23
            w1 = vdupq_n_f32(wc0[c + 4]);   // w1, v24
            w2 = vdupq_n_f32(wc0[c + 8]);   // w2, v25
            w3 = vdupq_n_f32(wc0[c + 12]);  // w3, v26
            w4 = vdupq_n_f32(wc0[c + 16]);  // w4, v27
            w5 = vdupq_n_f32(wc0[c + 20]);  // w5, v28
            w6 = vdupq_n_f32(wc0[c + 24]);  // w6, v29
            w7 = vdupq_n_f32(wc0[c + 28]);  // w7, v30
            w8 = vdupq_n_f32(wc0[c + 32]);  // w8, v31

            r0 = inr0;
            r1 = inr1;
            r2 = inr2;
            r3 = inr3;
            r4 = inr4;

            cnt = w_loop;
            // clang-format off
            asm volatile(
                INIT_C1 COMPUTE_C1 RESULT_C1
                : [cnt] "+r"(cnt),
                  [r0] "+r"(r0),[r1] "+r"(r1),
                  [r2] "+r"(r2),[r3] "+r"(r3),
                  [r4] "+r"(r4),
                  [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1)
                : [w0] "w"(w0),[w1] "w"(w1),[w2] "w"(w2),
                  [w3] "w"(w3),[w4] "w"(w4),[w5] "w"(w5),
                  [w6] "w"(w6),[w7] "w"(w7),[w8] "w"(w8)
                : "cc","memory","v0","v1","v2","v3",
                  "v4","v5","v6","v7","v8","v9","v10","v11",
                  "v12","v13","v14","v15","v16","v21","v22");
            // clang-format on
            wc0 += 36;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
#else   // not __aarch64__
          float* ptr_out0 = pre_out0;
          float* ptr_out1 = pre_out1;
          //! get valid weights of current output channel
          float w_tmp[12] = {wc0[c],
                             wc0[c + 4],
                             wc0[c + 8],
                             0.f,
                             wc0[c + 12],
                             wc0[c + 16],
                             wc0[c + 20],
                             0.f,
                             wc0[c + 24],
                             wc0[c + 28],
                             wc0[c + 32],
                             0.f};
          float32x4_t w0 = vld1q_f32(w_tmp);      // w0, w1, w2, q0
          float32x4_t w1 = vld1q_f32(w_tmp + 4);  // w3, w4, w5, q1
          float32x4_t w2 = vld1q_f32(w_tmp + 8);  // w6, w7, w8, q2

          const float* r0 = inr0;
          const float* r1 = inr1;
          const float* r2 = inr2;
          const float* r3 = inr3;
          const float* r4 = inr4;

          int cnt = w_loop / 2;

          if (cnt > 0) {
            // clang-format off
            asm volatile(
              INIT_C1_FIRST COMPUTE_C1
              : [cnt] "+r"(cnt),
                [r0] "+r"(r0),[r1] "+r"(r1),[r2] "+r"(r2),
                [r3] "+r"(r3),[r4] "+r"(r4),
                [ptr_out0] "+r"(ptr_out0),
                [ptr_out1] "+r"(ptr_out1)
              : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2)
              : "cc","memory","q3","q4",
                "q5","q6","q7","q8","q9","q10",
                "q11","q12","q13","q14","q15"
            );
          }
          //! deal with remain ow
          if (w_loop & 1) {
              ptr_out0[0] =
                  r0[0] * w_tmp[0] + r0[1] * w_tmp[1] + r0[2] * w_tmp[2] +
                  r1[0] * w_tmp[4] + r1[1] * w_tmp[5] + r1[2] * w_tmp[6] +
                  r2[0] * w_tmp[8] + r2[1] * w_tmp[9] + r2[2] * w_tmp[10];

              ptr_out0[1] =
                  r0[2] * w_tmp[0] + r0[3] * w_tmp[1] + r0[4] * w_tmp[2] +
                  r1[2] * w_tmp[4] + r1[3] * w_tmp[5] + r1[4] * w_tmp[6] +
                  r2[2] * w_tmp[8] + r2[3] * w_tmp[9] + r2[4] * w_tmp[10];

              ptr_out0[2] =
                  r0[4] * w_tmp[0] + r0[5] * w_tmp[1] + r0[6] * w_tmp[2] +
                  r1[4] * w_tmp[4] + r1[5] * w_tmp[5] + r1[6] * w_tmp[6] +
                  r2[4] * w_tmp[8] + r2[5] * w_tmp[9] + r2[6] * w_tmp[10];

              ptr_out0[3] =
                  r0[6] * w_tmp[0] + r0[7] * w_tmp[1] + r0[8] * w_tmp[2] +
                  r1[6] * w_tmp[4] + r1[7] * w_tmp[5] + r1[8] * w_tmp[6] +
                  r2[6] * w_tmp[8] + r2[7] * w_tmp[9] + r2[8] * w_tmp[10];

              ptr_out1[0] =
                  r2[0] * w_tmp[0] + r2[1] * w_tmp[1] + r2[2] * w_tmp[2] +
                  r3[0] * w_tmp[4] + r3[1] * w_tmp[5] + r3[2] * w_tmp[6] +
                  r4[0] * w_tmp[8] + r4[1] * w_tmp[9] + r4[2] * w_tmp[10];

              ptr_out1[1] =
                  r2[2] * w_tmp[0] + r2[3] * w_tmp[1] + r2[4] * w_tmp[2] +
                  r3[2] * w_tmp[4] + r3[3] * w_tmp[5] + r3[4] * w_tmp[6] +
                  r4[2] * w_tmp[8] + r4[3] * w_tmp[9] + r4[4] * w_tmp[10];

              ptr_out1[2] =
                  r2[4] * w_tmp[0] + r2[5] * w_tmp[1] + r2[6] * w_tmp[2] +
                  r3[4] * w_tmp[4] + r3[5] * w_tmp[5] + r3[6] * w_tmp[6] +
                  r4[4] * w_tmp[8] + r4[5] * w_tmp[9] + r4[6] * w_tmp[10];

              ptr_out1[3] =
                  r2[6] * w_tmp[0] + r2[7] * w_tmp[1] + r2[8] * w_tmp[2] +
                  r3[6] * w_tmp[4] + r3[7] * w_tmp[5] + r3[8] * w_tmp[6] +
                  r4[6] * w_tmp[8] + r4[7] * w_tmp[9] + r4[8] * w_tmp[10];
          }
          wc0 += 36;
          inr0 += win_round;
          inr1 += win_round;
          inr2 += win_round;
          inr3 += win_round;
          inr4 += win_round;
          for (int i = 0; i < ic - 1; ++i) {
            ptr_out0 = pre_out0;
            ptr_out1 = pre_out1;

            //! get valid weights of current output channel
            float w_tmp[12] = {wc0[c],
                               wc0[c + 4],
                               wc0[c + 8],
                               0.f,
                               wc0[c + 12],
                               wc0[c + 16],
                               wc0[c + 20],
                               0.f,
                               wc0[c + 24],
                               wc0[c + 28],
                               wc0[c + 32],
                               0.f};
            w0 = vld1q_f32(w_tmp);      // w0, w1, w2, q0
            w1 = vld1q_f32(w_tmp + 4);  // w3, w4, w5, q1
            w2 = vld1q_f32(w_tmp + 8);  // w6, w7, w8, q2

            r0 = inr0;
            r1 = inr1;
            r2 = inr2;
            r3 = inr3;
            r4 = inr4;

            cnt = w_loop / 2;
            if (cnt > 0) {
              // clang-format off
              asm volatile(
                INIT_C1 COMPUTE_C1
                : [cnt] "+r"(cnt),
                  [r0] "+r"(r0),[r1] "+r"(r1),[r2] "+r"(r2),
                  [r3] "+r"(r3),[r4] "+r"(r4),
                  [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1)
                : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2)
                : "cc","memory","q3","q4",
                  "q5","q6","q7","q8","q9","q10",
                  "q11","q12","q13","q14","q15"
             );
              // clang-format on
            }
            //! deal with remain ow
            if (w_loop & 1) {
              ptr_out0[0] +=
                  r0[0] * w_tmp[0] + r0[1] * w_tmp[1] + r0[2] * w_tmp[2] +
                  r1[0] * w_tmp[4] + r1[1] * w_tmp[5] + r1[2] * w_tmp[6] +
                  r2[0] * w_tmp[8] + r2[1] * w_tmp[9] + r2[2] * w_tmp[10];

              ptr_out0[1] +=
                  r0[2] * w_tmp[0] + r0[3] * w_tmp[1] + r0[4] * w_tmp[2] +
                  r1[2] * w_tmp[4] + r1[3] * w_tmp[5] + r1[4] * w_tmp[6] +
                  r2[2] * w_tmp[8] + r2[3] * w_tmp[9] + r2[4] * w_tmp[10];

              ptr_out0[2] +=
                  r0[4] * w_tmp[0] + r0[5] * w_tmp[1] + r0[6] * w_tmp[2] +
                  r1[4] * w_tmp[4] + r1[5] * w_tmp[5] + r1[6] * w_tmp[6] +
                  r2[4] * w_tmp[8] + r2[5] * w_tmp[9] + r2[6] * w_tmp[10];

              ptr_out0[3] +=
                  r0[6] * w_tmp[0] + r0[7] * w_tmp[1] + r0[8] * w_tmp[2] +
                  r1[6] * w_tmp[4] + r1[7] * w_tmp[5] + r1[8] * w_tmp[6] +
                  r2[6] * w_tmp[8] + r2[7] * w_tmp[9] + r2[8] * w_tmp[10];

              ptr_out1[0] +=
                  r2[0] * w_tmp[0] + r2[1] * w_tmp[1] + r2[2] * w_tmp[2] +
                  r3[0] * w_tmp[4] + r3[1] * w_tmp[5] + r3[2] * w_tmp[6] +
                  r4[0] * w_tmp[8] + r4[1] * w_tmp[9] + r4[2] * w_tmp[10];

              ptr_out1[1] +=
                  r2[2] * w_tmp[0] + r2[3] * w_tmp[1] + r2[4] * w_tmp[2] +
                  r3[2] * w_tmp[4] + r3[3] * w_tmp[5] + r3[4] * w_tmp[6] +
                  r4[2] * w_tmp[8] + r4[3] * w_tmp[9] + r4[4] * w_tmp[10];

              ptr_out1[2] +=
                  r2[4] * w_tmp[0] + r2[5] * w_tmp[1] + r2[6] * w_tmp[2] +
                  r3[4] * w_tmp[4] + r3[5] * w_tmp[5] + r3[6] * w_tmp[6] +
                  r4[4] * w_tmp[8] + r4[5] * w_tmp[9] + r4[6] * w_tmp[10];

              ptr_out1[3] +=
                  r2[6] * w_tmp[0] + r2[7] * w_tmp[1] + r2[8] * w_tmp[2] +
                  r3[6] * w_tmp[4] + r3[7] * w_tmp[5] + r3[8] * w_tmp[6] +
                  r4[6] * w_tmp[8] + r4[7] * w_tmp[9] + r4[8] * w_tmp[10];
            }

            wc0 += 36;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
#endif  // __aarch64__
          block_inr0 = block_inr4;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }
        write_to_output_c1_fp32(pre_out,
                                dout_batch,
                                c + c_round_down,
                                c + c_round_down + 1,
                                h,
                                h + h_kernel,
                                0,
                                wout_round,
                                oc,
                                oh,
                                ow,
                                false,
                                nullptr,
                                &act_param,
                                bias_ptr);
      }
      LITE_PARALLEL_END();
    }
  }
}

#ifdef __aarch64__
#else
#define FMLA_W00                                 \
  "vmla.f32   q15, q9, d12[0]          @ mul \n" \
  "vmla.f32   q12, q9, d0[0]           @ mul \n" \
  "vmla.f32   q13, q9, d4[0]           @ mul \n" \
  "vmla.f32   q14, q9, d8[0]           @ mul \n" \
  "vld1.32  {d18-d19}, [%[wc0]]!       @ load w2, w3\n"
#define FMLA_W01                                  \
  "vmla.f32   q15, q10, d12[1]          @ mul \n" \
  "vmla.f32   q12, q10, d0[1]           @ mul \n" \
  "vmla.f32   q13, q10, d4[1]           @ mul \n" \
  "vmla.f32   q14, q10, d8[1]           @ mul \n" \
  "vld1.32  {d20-d21}, [%[wc0]]!        @ load w2, w3\n"
#define FMLA_W02                                  \
  "vmla.f32   q15, q11, d13[0]          @ mul \n" \
  "vmla.f32   q12, q11, d1[0]           @ mul \n" \
  "vmla.f32   q13, q11, d5[0]           @ mul \n" \
  "vmla.f32   q14, q11, d9[0]           @ mul \n" \
  "vld1.32  {d22-d23}, [%[wc0]]!        @ load w2, w3\n"
#define FMLA_W10                                 \
  "vmla.f32   q15, q9, d14[0]          @ mul \n" \
  "vmla.f32   q12, q9, d2[0]           @ mul \n" \
  "vmla.f32   q13, q9, d6[0]           @ mul \n" \
  "vmla.f32   q14, q9, d10[0]          @ mul \n" \
  "vld1.32  {d18-d19}, [%[wc0]]!       @ load w2, w3\n"
#define FMLA_W11                                  \
  "vmla.f32   q15, q10, d14[1]          @ mul \n" \
  "vmla.f32   q12, q10, d2[1]           @ mul \n" \
  "vmla.f32   q13, q10, d6[1]           @ mul \n" \
  "vmla.f32   q14, q10, d10[1]          @ mul \n" \
  "vld1.32  {d20-d21}, [%[wc0]]!        @ load w2, w3\n"
#define FMLA_W12                                  \
  "vmla.f32   q15, q11, d15[0]          @ mul \n" \
  "vmla.f32   q12, q11, d3[0]           @ mul \n" \
  "vmla.f32   q13, q11, d7[0]           @ mul \n" \
  "vmla.f32   q14, q11, d11[0]          @ mul \n" \
  "vld1.32  {d22-d23}, [%[wc0]]!        @ load w2, w3\n"
#define FMLA_W20                                 \
  "vmla.f32   q15, q9, d0[0]           @ mul \n" \
  "vmla.f32   q12, q9, d4[0]           @ mul \n" \
  "vmla.f32   q13, q9, d8[0]           @ mul \n" \
  "vmla.f32   q14, q9, d12[0]          @ mul \n" \
  "vld1.32  {d18-d19}, [%[wc0]]!       @ load w2, w3\n"
#define FMLA_W21                                  \
  "vmla.f32   q15, q10, d0[1]           @ mul \n" \
  "vmla.f32   q12, q10, d4[1]           @ mul \n" \
  "vmla.f32   q13, q10, d8[1]           @ mul \n" \
  "vmla.f32   q14, q10, d12[1]          @ mul \n" \
  "vld1.32  {d20-d21}, [%[wc0]]!        @ load w2, w3\n"
#define FMLA_W22                                  \
  "vmla.f32   q15, q11, d1[0]           @ mul \n" \
  "vmla.f32   q12, q11, d5[0]           @ mul \n" \
  "vmla.f32   q13, q11, d9[0]           @ mul \n" \
  "vmla.f32   q14, q11, d13[0]          @ mul \n" \
  "vld1.32  {d22-d23}, [%[wc0]]!        @ load w2, w3\n"
#endif
void conv_3x3s2_direct_fp32_c3(const float* i_data,
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
                               ARMContext* ctx) {
  //! 3x3s2 convolution, implemented by NHWC4 algorithm
  //! prepack input to tmp buffer NHWC4
  //! weights layout [oc, ic, kh, kw] transform to [oc/4, kh, kw, ic, 4]
  //! write output to tmp buffer
  auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  bool flag_bias = (param.bias != nullptr);
  auto out_h_block = 1;
  int in_channel = 4;
  const int threads = ctx->threads();
  int l2_size = ctx->llc_size() / sizeof(float);
  const int pad_w = paddings[2];
  const int pad_h = paddings[0];
  const int wout_round = ROUNDUP(ow, OUT_W_BLOCK);
  const int win_round = wout_round * 2 /*stride_w*/ + 1;

  //! get h block
  //! win_round * ic * hin_r_block + wout_round * OUT_C_BLOCK * hout_r_block
  //! * threads = l2_size
  //! win_round = 2 * wout_round + 1
  //! hin_r_block = 2 * hout_r_block + 1
  int hout_r_block =
      (l2_size - 2 * wout_round * in_channel - in_channel) /
      ((4 * wout_round + 2) * in_channel + wout_round * OUT_C_BLOCK * threads);
  hout_r_block = hout_r_block > oh ? oh : hout_r_block;
  hout_r_block = (hout_r_block / OUT_H_BLOCK) * OUT_H_BLOCK;
  hout_r_block = hout_r_block < OUT_H_BLOCK ? OUT_H_BLOCK : hout_r_block;

  const int hin_r_block = hout_r_block * 2 /*stride_h*/ + 1;

  int in_len = win_round * in_channel;
  int pre_in_size = hin_r_block * in_len;
  int pre_out_size = OUT_C_BLOCK * hout_r_block * wout_round;

  float* tmp_work_space = ctx->workspace_data<float>();
  float ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(float) * win_round);

  //! l2_cache start
  float* pre_din = tmp_work_space;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;
  int w_stride = ic * 9; /*kernel_w * kernel_h*/
  int w_stride_chin = OUT_C_BLOCK * 9;

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int c_round_down = ROUNDUP(oc, OUT_W_BLOCK);

  int out_row_stride = OUT_C_BLOCK * wout_round;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  for (int n = 0; n < bs; ++n) {
    const float* din_batch = i_data + n * ic * size_in_channel;
    float* dout_batch = o_data + n * oc * size_out_channel;
    for (int h = 0; h < oh; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > oh) {
        h_kernel = oh - h;
      }

      int hs = h * 2 /*stride_h*/ - pad_h;
      int he = hs + h_kernel * 2 /*stride_h*/ + 1;

      prepack_input_nxwc4_dw(
          din_batch, pre_din, 0, hs, he, ws, we, ic, win, ih, ptr_zero);

      const float* cblock_inr0 = pre_din;
      const float* cblock_inr1 = cblock_inr0 + in_len;
      const float* cblock_inr2 = cblock_inr1 + in_len;
      const float* cblock_inr3 = cblock_inr2 + in_len;
      const float* cblock_inr4 = cblock_inr3 + in_len;
      LITE_PARALLEL_COMMON_BEGIN(c, tid, c_round_down, 0, OUT_C_BLOCK) {
#ifdef LITE_USE_THREAD_POOL
        float* pre_out = pre_din + pre_in_size + tid * pre_out_size;
#elif defined(ARM_WITH_OMP)
        float* pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float* pre_out = pre_din + pre_in_size;
#endif
        const float* block_inr0 = cblock_inr0;
        const float* block_inr1 = cblock_inr1;
        const float* block_inr2 = cblock_inr2;
        const float* block_inr3 = cblock_inr3;
        const float* block_inr4 = cblock_inr4;

        const float* weight_c = weights + c * w_stride;
        const float* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = bias + c;
        }

#ifdef __aarch64__
        for (int hk = 0; hk < h_kernel; hk += OUT_H_BLOCK) {
          const float* wc0 = weight_c;

          const float* inr0 = block_inr0;
          const float* inr1 = block_inr1;
          const float* inr2 = block_inr2;
          const float* inr3 = block_inr3;
          const float* inr4 = block_inr4;

          float* pre_out0 = pre_out + hk * out_row_stride;
          float* pre_out1 = pre_out0 + out_row_stride;
          int cnt = w_loop;
          const float* r0 = inr0;
          const float* r1 = inr1;
          const float* r2 = inr2;
          const float* r3 = inr3;
          const float* r4 = inr4;
          const float* wc00 = wc0;
          float* ptr_out0 = pre_out0;
          float* ptr_out1 = pre_out1;

          // clang-format off
          asm volatile(
            "ldp    q0, q1,   [%[r0]], #32\n"
            "ldp    q9, q10,   [%[wc]], #32\n"
            "ldp    q19, q20,   [%[r2]], #32\n"
            "ldp    q2, q3,   [%[r0]], #32\n"
            "ldp    q21, q22,   [%[r2]], #32\n"
            "ldp    q11, q12,   [%[wc]], #32\n"
            "ldp    q4, q5,   [%[r0]], #32\n"
            "ldp    q23, q24,   [%[r2]], #32\n"
            "1:      \n"
            /* line 0 */
            // compute zero i00-i20
            "ldp    q6, q7,   [%[r0]], #32\n"
            "ldp    q25, q26,   [%[r2]], #32\n"
            "fmul   v15.4s,  v9.4s,  v0.s[0]\n"
            "fmul   v16.4s,  v9.4s,  v2.s[0]\n"
            "fmul   v17.4s,  v9.4s,  v4.s[0]\n"
            "fmul   v18.4s,  v9.4s,  v6.s[0]\n"
            "fmul   v28.4s,  v9.4s,  v19.s[0]\n"
            "fmul   v29.4s,  v9.4s,  v21.s[0]\n"
            "fmul   v30.4s,  v9.4s,  v23.s[0]\n"
            "fmul   v31.4s,  v9.4s,  v25.s[0]\n"
            "ldr    q8,   [%[r0]]\n"
            "ldr    q27,   [%[r2]]\n"
            "fmla   v15.4s,  v10.4s,  v0.s[1]\n"
            "fmla   v16.4s,  v10.4s,  v2.s[1]\n"
            "fmla   v17.4s,  v10.4s,  v4.s[1]\n"
            "fmla   v18.4s,  v10.4s,  v6.s[1]\n"
            "fmla   v28.4s,  v10.4s,  v19.s[1]\n"
            "fmla   v29.4s,  v10.4s,  v21.s[1]\n"
            "fmla   v30.4s,  v10.4s,  v23.s[1]\n"
            "fmla   v31.4s,  v10.4s,  v25.s[1]\n"
            "ldp    q13, q14,   [%[wc]], #32\n"
            "fmla   v15.4s,  v11.4s,  v0.s[2]\n"
            "fmla   v16.4s,  v11.4s,  v2.s[2]\n"
            "fmla   v17.4s,  v11.4s,  v4.s[2]\n"
            "fmla   v18.4s,  v11.4s,  v6.s[2]\n"
            "fmla   v28.4s,  v11.4s,  v19.s[2]\n"
            "fmla   v29.4s,  v11.4s,  v21.s[2]\n"
            "fmla   v30.4s,  v11.4s,  v23.s[2]\n"
            "fmla   v31.4s,  v11.4s,  v25.s[2]\n"
            // compute one i01-i21
            "ldp    q9, q10, [%[wc]], #32\n"
            "fmla   v15.4s,  v12.4s,  v1.s[0]\n"
            "fmla   v16.4s,  v12.4s,  v3.s[0]\n"
            "fmla   v17.4s,  v12.4s,  v5.s[0]\n"
            "fmla   v18.4s,  v12.4s,  v7.s[0]\n"
            "fmla   v28.4s,  v12.4s,  v20.s[0]\n"
            "fmla   v29.4s,  v12.4s,  v22.s[0]\n"
            "fmla   v30.4s,  v12.4s,  v24.s[0]\n"
            "fmla   v31.4s,  v12.4s,  v26.s[0]\n"
            "ldp   q11, q12, [%[wc]], #32\n"

            "fmla   v15.4s,  v13.4s,  v1.s[1]\n"
            "fmla   v16.4s,  v13.4s,  v3.s[1]\n"
            "fmla   v17.4s,  v13.4s,  v5.s[1]\n"
            "fmla   v18.4s,  v13.4s,  v7.s[1]\n"
            "fmla   v28.4s,  v13.4s,  v20.s[1]\n"
            "fmla   v29.4s,  v13.4s,  v22.s[1]\n"
            "fmla   v30.4s,  v13.4s,  v24.s[1]\n"
            "fmla   v31.4s,  v13.4s,  v26.s[1]\n"

            "fmla   v15.4s,  v14.4s,  v1.s[2]\n"
            "fmla   v16.4s,  v14.4s,  v3.s[2]\n"
            "fmla   v17.4s,  v14.4s,  v5.s[2]\n"
            "fmla   v18.4s,  v14.4s,  v7.s[2]\n"
            "fmla   v28.4s,  v14.4s,  v20.s[2]\n"
            "fmla   v29.4s,  v14.4s,  v22.s[2]\n"
            "fmla   v30.4s,  v14.4s,  v24.s[2]\n"
            "fmla   v31.4s,  v14.4s,  v26.s[2]\n"
            "ldp    q0, q1,   [%[r1]], #32\n"
            "ldp    q13, q14,   [%[wc]], #32\n"
            // compute two i02-i22
            "fmla   v15.4s,  v9.4s,  v2.s[0]\n"
            "fmla   v16.4s,  v9.4s,  v4.s[0]\n"
            "fmla   v17.4s,  v9.4s,  v6.s[0]\n"
            "fmla   v18.4s,  v9.4s,  v8.s[0]\n"
            "fmla   v28.4s,  v9.4s,  v21.s[0]\n"
            "fmla   v29.4s,  v9.4s,  v23.s[0]\n"
            "fmla   v30.4s,  v9.4s,  v25.s[0]\n"
            "fmla   v31.4s,  v9.4s,  v27.s[0]\n"

            "fmla   v15.4s,  v10.4s,  v2.s[1]\n"
            "fmla   v16.4s,  v10.4s,  v4.s[1]\n"
            "fmla   v17.4s,  v10.4s,  v6.s[1]\n"
            "fmla   v18.4s,  v10.4s,  v8.s[1]\n"
            "fmla   v28.4s,  v10.4s,  v21.s[1]\n"
            "fmla   v29.4s,  v10.4s,  v23.s[1]\n"
            "fmla   v30.4s,  v10.4s,  v25.s[1]\n"
            "fmla   v31.4s,  v10.4s,  v27.s[1]\n"
            "ldp    q9, q10,   [%[wc]], #32\n"

            "fmla   v15.4s,  v11.4s,  v2.s[2]\n"
            "fmla   v16.4s,  v11.4s,  v4.s[2]\n"
            "ldp    q2, q3,   [%[r1]], #32\n"
            "fmla   v17.4s,  v11.4s,  v6.s[2]\n"
            "fmla   v18.4s,  v11.4s,  v8.s[2]\n"
            "ldp    q4, q5,   [%[r1]], #32\n"
            "fmla   v28.4s,  v11.4s,  v21.s[2]\n"
            "fmla   v29.4s,  v11.4s,  v23.s[2]\n"
            "fmla   v30.4s,  v11.4s,  v25.s[2]\n"
            "fmla   v31.4s,  v11.4s,  v27.s[2]\n"
            "ldp    q6, q7,   [%[r1]], #32\n"
            /* line 1 */
            "fmla   v15.4s,  v12.4s,  v0.s[0]\n"
            "fmla   v16.4s,  v12.4s,  v2.s[0]\n"
            "fmla   v17.4s,  v12.4s,  v4.s[0]\n"
            "fmla   v18.4s,  v12.4s,  v6.s[0]\n"
            "ldp    q11, q12,   [%[wc]], #32\n"
            "fmla   v15.4s,  v13.4s,  v0.s[1]\n"
            "fmla   v16.4s,  v13.4s,  v2.s[1]\n"
            "fmla   v17.4s,  v13.4s,  v4.s[1]\n"
            "fmla   v18.4s,  v13.4s,  v6.s[1]\n"
            "ldr    q8,   [%[r1]]\n"
            "fmla   v15.4s,  v14.4s,  v0.s[2]\n"
            "fmla   v16.4s,  v14.4s,  v2.s[2]\n"
            "fmla   v17.4s,  v14.4s,  v4.s[2]\n"
            "fmla   v18.4s,  v14.4s,  v6.s[2]\n"
            "ldp    q13, q14,   [%[wc]], #32\n"

            "fmla   v15.4s,  v9.4s,  v1.s[0]\n"
            "fmla   v16.4s,  v9.4s,  v3.s[0]\n"
            "fmla   v17.4s,  v9.4s,  v5.s[0]\n"
            "fmla   v18.4s,  v9.4s,  v7.s[0]\n"
            "fmla   v15.4s,  v10.4s,  v1.s[1]\n"
            "fmla   v16.4s,  v10.4s,  v3.s[1]\n"
            "fmla   v17.4s,  v10.4s,  v5.s[1]\n"
            "fmla   v18.4s,  v10.4s,  v7.s[1]\n"
            "fmla   v15.4s,  v11.4s,  v1.s[2]\n"
            "fmla   v16.4s,  v11.4s,  v3.s[2]\n"
            "fmla   v17.4s,  v11.4s,  v5.s[2]\n"
            "fmla   v18.4s,  v11.4s,  v7.s[2]\n"
            "ldp    q0, q1,   [%[r3]], #32\n"
            "sub    %[wc], %[wc], #144\n"
            
            "fmla   v15.4s,  v12.4s,  v2.s[0]\n"
            "fmla   v16.4s,  v12.4s,  v4.s[0]\n"
            "fmla   v17.4s,  v12.4s,  v6.s[0]\n"
            "fmla   v18.4s,  v12.4s,  v8.s[0]\n"
            "fmla   v15.4s,  v13.4s,  v2.s[1]\n"
            "fmla   v16.4s,  v13.4s,  v4.s[1]\n"
            "fmla   v17.4s,  v13.4s,  v6.s[1]\n"
            "fmla   v18.4s,  v13.4s,  v8.s[1]\n"
            "fmla   v15.4s,  v14.4s,  v2.s[2]\n"
            "fmla   v16.4s,  v14.4s,  v4.s[2]\n"
            "ldp    q2, q3,   [%[r3]], #32\n"
            "fmla   v17.4s,  v14.4s,  v6.s[2]\n"
            "fmla   v18.4s,  v14.4s,  v8.s[2]\n"
            "ldp    q4, q5,   [%[r3]], #32\n"
            "ldp    q6, q7,   [%[r3]], #32\n"

            "fmla   v28.4s,  v9.4s, v1.s[0]\n"
            "fmla   v29.4s,  v9.4s, v3.s[0]\n"
            "ldr    q8,   [%[r3]]\n"
            "fmla   v30.4s,  v9.4s, v5.s[0]\n"
            "fmla   v31.4s,  v9.4s, v7.s[0]\n"
            "fmla   v28.4s,  v10.4s, v1.s[1]\n"
            "fmla   v29.4s,  v10.4s, v3.s[1]\n"
            "fmla   v30.4s,  v10.4s, v5.s[1]\n"
            "fmla   v31.4s,  v10.4s, v7.s[1]\n"
            "ldp    q9, q10,   [%[wc]], #32\n"
            "fmla   v28.4s,  v11.4s, v1.s[2]\n"
            "fmla   v29.4s,  v11.4s, v3.s[2]\n"
            "fmla   v30.4s,  v11.4s, v5.s[2]\n"
            "fmla   v31.4s,  v11.4s, v7.s[2]\n"

            "fmla   v28.4s,  v12.4s, v2.s[0]\n"
            "fmla   v29.4s,  v12.4s, v4.s[0]\n"
            "fmla   v30.4s,  v12.4s, v6.s[0]\n"
            "fmla   v31.4s,  v12.4s, v8.s[0]\n"
            "ldp    q11, q12,   [%[wc]], #32\n"
            "fmla   v28.4s,  v13.4s, v2.s[1]\n"
            "fmla   v29.4s,  v13.4s, v4.s[1]\n"
            "fmla   v30.4s,  v13.4s, v6.s[1]\n"
            "fmla   v31.4s,  v13.4s, v8.s[1]\n"
            "fmla   v28.4s,  v14.4s, v2.s[2]\n"
            "fmla   v29.4s,  v14.4s, v4.s[2]\n"
            "fmla   v30.4s,  v14.4s, v6.s[2]\n"
            "fmla   v31.4s,  v14.4s, v8.s[2]\n"
            "add    %[wc], %[wc], #80\n"

            "fmla   v28.4s,  v9.4s, v0.s[0]\n"
            "fmla   v29.4s,  v9.4s, v2.s[0]\n"
            "fmla   v30.4s,  v9.4s, v4.s[0]\n"
            "fmla   v31.4s,  v9.4s, v6.s[0]\n"
            "ldp    q13, q14,   [%[wc]], #32\n" // line 2 w0-w2
            "fmla   v28.4s,  v10.4s, v0.s[1]\n"
            "fmla   v29.4s,  v10.4s, v2.s[1]\n"
            "fmla   v30.4s,  v10.4s, v4.s[1]\n"
            "fmla   v31.4s,  v10.4s, v6.s[1]\n"
            "fmla   v28.4s,  v11.4s, v0.s[2]\n"
            "fmla   v29.4s,  v11.4s, v2.s[2]\n"
            "ldp    q0, q1,   [%[r4]], #32\n"
            "fmla   v30.4s,  v11.4s, v4.s[2]\n"
            "fmla   v31.4s,  v11.4s, v6.s[2]\n"
            "ldp    q2, q3,   [%[r4]], #32\n"
            "ldp    q9, q10,   [%[wc]], #32\n"
            /* line 2 */
            "ldp    q4, q5,   [%[r4]], #32\n"
            "fmla   v15.4s,  v13.4s,  v19.s[0]\n"
            "fmla   v16.4s,  v13.4s,  v21.s[0]\n"
            "fmla   v17.4s,  v13.4s,  v23.s[0]\n"
            "fmla   v18.4s,  v13.4s,  v25.s[0]\n"
            "ldp    q6, q7,   [%[r4]], #32\n"
            "ldp    q11, q12,   [%[wc]], #32\n"
            "fmla   v28.4s,  v13.4s,  v0.s[0]\n"
            "fmla   v29.4s,  v13.4s,  v2.s[0]\n"
            "fmla   v30.4s,  v13.4s,  v4.s[0]\n"
            "fmla   v31.4s,  v13.4s,  v6.s[0]\n"
            "ldr    q8,   [%[r4]]\n"

            "fmla   v15.4s,  v14.4s,  v19.s[1]\n"
            "fmla   v16.4s,  v14.4s,  v21.s[1]\n"
            "fmla   v17.4s,  v14.4s,  v23.s[1]\n"
            "fmla   v18.4s,  v14.4s,  v25.s[1]\n"
            "fmla   v28.4s,  v14.4s,  v0.s[1]\n"
            "fmla   v29.4s,  v14.4s,  v2.s[1]\n"
            "fmla   v30.4s,  v14.4s,  v4.s[1]\n"
            "fmla   v31.4s,  v14.4s,  v6.s[1]\n"
            "ldp    q13, q14,   [%[wc]], #32\n"
            "fmla   v15.4s,  v9.4s,  v19.s[2]\n"
            "fmla   v16.4s,  v9.4s,  v21.s[2]\n"
            "fmla   v17.4s,  v9.4s,  v23.s[2]\n"
            "fmla   v18.4s,  v9.4s,  v25.s[2]\n"
            "fmla   v28.4s,  v9.4s,  v0.s[2]\n"
            "fmla   v29.4s,  v9.4s,  v2.s[2]\n"
            "fmla   v30.4s,  v9.4s,  v4.s[2]\n"
            "fmla   v31.4s,  v9.4s,  v6.s[2]\n"

            "fmla   v15.4s,  v10.4s,  v20.s[0]\n"
            "fmla   v16.4s,  v10.4s,  v22.s[0]\n"
            "fmla   v17.4s,  v10.4s,  v24.s[0]\n"
            "fmla   v18.4s,  v10.4s,  v26.s[0]\n"
            "fmla   v28.4s,  v10.4s,  v1.s[0]\n"
            "fmla   v29.4s,  v10.4s,  v3.s[0]\n"
            "fmla   v30.4s,  v10.4s,  v5.s[0]\n"
            "fmla   v31.4s,  v10.4s,  v7.s[0]\n"
            "ldr    q9,   [%[wc]], #16\n"

            "fmla   v15.4s,  v11.4s,  v20.s[1]\n"
            "fmla   v16.4s,  v11.4s,  v22.s[1]\n"
            "fmla   v17.4s,  v11.4s,  v24.s[1]\n"
            "fmla   v18.4s,  v11.4s,  v26.s[1]\n"
            "fmla   v28.4s,  v11.4s,  v1.s[1]\n"
            "fmla   v29.4s,  v11.4s,  v3.s[1]\n"
            "fmla   v30.4s,  v11.4s,  v5.s[1]\n"
            "fmla   v31.4s,  v11.4s,  v7.s[1]\n"

            "fmla   v15.4s,  v12.4s,  v20.s[2]\n"
            "fmla   v16.4s,  v12.4s,  v22.s[2]\n"
            "fmla   v17.4s,  v12.4s,  v24.s[2]\n"
            "fmla   v18.4s,  v12.4s,  v26.s[2]\n"
            "fmla   v28.4s,  v12.4s,  v1.s[2]\n"
            "fmla   v29.4s,  v12.4s,  v3.s[2]\n"
            "fmla   v30.4s,  v12.4s,  v5.s[2]\n"
            "fmla   v31.4s,  v12.4s,  v7.s[2]\n"

            "sub %[wc], %[wc], #432\n"
            "fmla   v15.4s,  v13.4s,  v21.s[0]\n"
            "fmla   v16.4s,  v13.4s,  v23.s[0]\n"
            "fmla   v17.4s,  v13.4s,  v25.s[0]\n"
            "fmla   v18.4s,  v13.4s,  v27.s[0]\n"
            "fmla   v28.4s,  v13.4s,  v2.s[0]\n"
            "fmla   v29.4s,  v13.4s,  v4.s[0]\n"
            "fmla   v30.4s,  v13.4s,  v6.s[0]\n"
            "fmla   v31.4s,  v13.4s,  v8.s[0]\n"

            "fmla   v15.4s,  v14.4s,  v21.s[1]\n"
            "fmla   v16.4s,  v14.4s,  v23.s[1]\n"
            "fmla   v17.4s,  v14.4s,  v25.s[1]\n"
            "fmla   v18.4s,  v14.4s,  v27.s[1]\n"
            "fmla   v28.4s,  v14.4s,  v2.s[1]\n"
            "fmla   v29.4s,  v14.4s,  v4.s[1]\n"
            "fmla   v30.4s,  v14.4s,  v6.s[1]\n"
            "fmla   v31.4s,  v14.4s,  v8.s[1]\n"

            "ldp    q0, q1,   [%[r0]], #32\n"
            "fmla   v15.4s,  v9.4s,  v21.s[2]\n"
            "fmla   v16.4s,  v9.4s,  v23.s[2]\n"
            "fmla   v17.4s,  v9.4s,  v25.s[2]\n"
            "fmla   v18.4s,  v9.4s,  v27.s[2]\n"
            "ldp    q19, q20,   [%[r2]], #32\n"
            "fmla   v28.4s,  v9.4s,  v2.s[2]\n"
            "fmla   v29.4s,  v9.4s,  v4.s[2]\n"
            "fmla   v30.4s,  v9.4s,  v6.s[2]\n"
            "fmla   v31.4s,  v9.4s,  v8.s[2]\n"

            "subs   %w[cnt], %w[cnt], #1\n"
            "stp    q15, q16,  [%[ptr_out0]], #32\n"
            "ldp    q2, q3,   [%[r0]], #32\n"
            "stp    q28, q29,  [%[ptr_out1]], #32\n"
            "ldp    q21, q22,   [%[r2]], #32\n"
            "ldp    q9, q10,   [%[wc]], #32\n"
            "stp    q17, q18,  [%[ptr_out0]], #32\n"
            "ldp    q4, q5,   [%[r0]], #32\n"
            "stp    q30, q31,  [%[ptr_out1]], #32\n"
            "ldp    q23, q24,   [%[r2]], #32\n"
            "ldp    q11, q12,   [%[wc]], #32\n"
            "bne    1b\n"
            : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1),
              [r2] "+r"(r2), [r3] "+r"(r3), [r4] "+r"(r4),
              [wc]"+r"(wc00), [ptr_out0] "+r"(ptr_out0), [ptr_out1] "+r"(ptr_out1)
            : 
            : "cc","memory","v0","v1","v2","v3","v4",
              "v5","v6","v7","v8","v9","v10","v11","v12","v13",
              "v14","v15","v16","v17","v18", "v19", "v20", "v21", "v22",
              "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
          // clang-format on
          block_inr0 = block_inr4;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }
#else   // not __aarch64__
        for (int hk = 0; hk < h_kernel; hk += out_h_block) {
          const float* wc0 = weight_c;

          const float* inr0 = block_inr0;
          const float* inr1 = block_inr1;
          const float* inr2 = block_inr2;

          float* pre_out0 = pre_out + hk * out_row_stride;
          int cnt = w_loop;
          const float* r0 = inr0;
          const float* r1 = inr1;
          const float* r2 = inr2;
          const float* wc00 = wc0;
          float* ptr_out0 = pre_out0;

          // clang-format off
          asm volatile(
            "vld1.32  {d0-d3}, [%[r0]]!          @ load q0, q1\n"
            "vld1.32  {d18-d19}, [%[wc0]]!       @ load w0, w1\n"
            "vld1.32  {d4-d7}, [%[r0]]!          @ load q2, q3\n"
            "vld1.32  {d20-d21}, [%[wc0]]!       @ load w2, w3\n"
            "vld1.32  {d8-d11}, [%[r0]]!          @ load q4, q5\n"
            "vld1.32  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"
            "1: \n"
            "vld1.32  {d12-d15}, [%[r0]]!          @ load q6, q7\n"
            /* line 0*/
            "vmul.f32   q15, q9, d12[0]          @ mul \n"
            "vmul.f32   q12, q9, d0[0]           @ mul \n"
            "vmul.f32   q13, q9, d4[0]           @ mul \n"
            "vmul.f32   q14, q9, d8[0]           @ mul \n"
            "vld1.32  {d18-d19}, [%[wc0]]!       @ load w2, w3\n"
            FMLA_W01
            FMLA_W02

            FMLA_W10
            FMLA_W11
            FMLA_W12
            "vld1.32  {d0-d1}, [%[r0]]          @ load q0, q1\n"

            FMLA_W20
            FMLA_W21
            "vmla.f32   q15, q11, d1[0]          @ mul \n"
            "vld1.32  {d0-d3}, [%[r1]]!          @ load q0, q1\n"
            "vmla.f32   q12, q11, d5[0]           @ mul \n"
            "vld1.32  {d4-d7}, [%[r1]]!          @ load q0, q1\n"
            "vmla.f32   q13, q11, d9[0]           @ mul \n"
            "vld1.32  {d8-d11}, [%[r1]]!          @ load q0, q1\n"
            "vmla.f32   q14, q11, d13[0]           @ mul \n"
            "vld1.32  {d12-d15}, [%[r1]]!          @ load q0, q1\n"
            "vld1.32  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"
            /* line 1*/
            FMLA_W00
            FMLA_W01
            FMLA_W02

            FMLA_W10
            FMLA_W11
            FMLA_W12
            "vld1.32  {d0-d1}, [%[r1]]          @ load q0, q1\n"

            FMLA_W20
            FMLA_W21
            "vmla.f32   q15, q11, d1[0]          @ mul \n"
            "vld1.32  {d0-d3}, [%[r2]]!          @ load q0, q1\n"
            "vmla.f32   q12, q11, d5[0]           @ mul \n"
            "vld1.32  {d4-d7}, [%[r2]]!          @ load q0, q1\n"
            "vmla.f32   q13, q11, d9[0]           @ mul \n"
            "vld1.32  {d8-d11}, [%[r2]]!          @ load q0, q1\n"
            "vmla.f32   q14, q11, d13[0]           @ mul \n"
            "vld1.32  {d12-d15}, [%[r2]]!          @ load q0, q1\n"
            "vld1.32  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"

            /* line 2*/
            FMLA_W00
            FMLA_W01
            FMLA_W02

            FMLA_W10
            FMLA_W11
            FMLA_W12
            "vld1.32  {d0-d1}, [%[r2]]          @ load q0, q1\n"

            "sub %[wc0], %[wc0], #432\n"
            FMLA_W20
            FMLA_W21
            FMLA_W22
            "subs %[cnt], #1\n"
            "vld1.32  {d0-d3}, [%[r0]]!          @ load q0, q1\n"
            "vst1.32    {d24-d27}, [%[ptr_out]]! @ store \n"
            "vld1.32  {d4-d7}, [%[r0]]!          @ load q2, q3\n"
            "vst1.32    {d28-d31}, [%[ptr_out]]! @ store \n"
            "vld1.32  {d8-d11}, [%[r0]]!          @ load q4, q5\n"
            "bne    1b\n"

          : [cnt] "+r"(cnt),
              [r0] "+r"(r0),[r1] "+r"(r1),
              [r2] "+r"(r2), [ptr_out] "+r"(ptr_out0), [wc0] "+r"(wc00)
          :
          : "cc","memory","q0","q1","q2","q3","q4",
              "q5","q6","q7","q8","q9","q10",
              "q11","q12","q13","q14","q15"
          );
          // clang-format on
          block_inr0 = block_inr2;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
        }
#endif  // __aarch64__
        write_to_output_c4_fp32(pre_out,
                                dout_batch,
                                c,
                                c + OUT_C_BLOCK,
                                h,
                                h + h_kernel,
                                0,
                                wout_round,
                                oc,
                                oh,
                                ow,
                                false,
                                nullptr,
                                &act_param,
                                bias_ptr);
      }
      LITE_PARALLEL_COMMON_END();
    }
  }
}

void conv_3x3s2_direct_fp32_c3_a53(const float* i_data,
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
                                   ARMContext* ctx) {
  //! 3x3s2 convolution, implemented by NHWC4 algorithm
  //! prepack input to tmp buffer NHWC4
  //! weights layout [oc, ic, kh, kw] transform to [oc/4, kh, kw, ic, 4]
  //! write output to tmp buffer
  auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  bool flag_bias = (param.bias != nullptr);
  auto out_h_block = 1;
  int in_channel = 4;
  const int threads = ctx->threads();
  int l2_size = ctx->llc_size() / sizeof(float);
  const int pad_w = paddings[2];
  const int pad_h = paddings[0];
  const int wout_round = ROUNDUP(ow, OUT_W_BLOCK);
  const int win_round = wout_round * 2 /*stride_w*/ + 1;

  //! get h block
  //! win_round * ic * hin_r_block + wout_round * OUT_C_BLOCK * hout_r_block
  //! * threads = l2_size
  //! win_round = 2 * wout_round + 1
  //! hin_r_block = 2 * hout_r_block + 1
  int hout_r_block =
      (l2_size - 2 * wout_round * in_channel - in_channel) /
      ((4 * wout_round + 2) * in_channel + wout_round * OUT_C_BLOCK * threads);
  hout_r_block = hout_r_block > oh ? oh : hout_r_block;
  hout_r_block = (hout_r_block / OUT_H_BLOCK) * OUT_H_BLOCK;
  hout_r_block = hout_r_block < OUT_H_BLOCK ? OUT_H_BLOCK : hout_r_block;

  const int hin_r_block = hout_r_block * 2 /*stride_h*/ + 1;

  int in_len = win_round * in_channel;
  int pre_in_size = hin_r_block * in_len;
  int pre_out_size = OUT_C_BLOCK * hout_r_block * wout_round;

  float* tmp_work_space = ctx->workspace_data<float>();
  float ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(float) * win_round);

  //! l2_cache start
  float* pre_din = tmp_work_space;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;
  int w_stride = ic * 9; /*kernel_w * kernel_h*/
  int w_stride_chin = OUT_C_BLOCK * 9;

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int c_round_down = ROUNDUP(oc, OUT_W_BLOCK);

  int out_row_stride = OUT_C_BLOCK * wout_round;
  auto act_type = act_param.active_type;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      alpha = act_param.Leaky_relu_alpha;
    }
  }
  for (int n = 0; n < bs; ++n) {
    const float* din_batch = i_data + n * ic * size_in_channel;
    float* dout_batch = o_data + n * oc * size_out_channel;
    for (int h = 0; h < oh; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > oh) {
        h_kernel = oh - h;
      }

      int hs = h * 2 /*stride_h*/ - pad_h;
      int he = hs + h_kernel * 2 /*stride_h*/ + 1;

      prepack_input_nxwc4_dw(
          din_batch, pre_din, 0, hs, he, ws, we, ic, win, ih, ptr_zero);

      const float* cblock_inr0 = pre_din;
      const float* cblock_inr1 = cblock_inr0 + in_len;
      const float* cblock_inr2 = cblock_inr1 + in_len;
      const float* cblock_inr3 = cblock_inr2 + in_len;
      const float* cblock_inr4 = cblock_inr3 + in_len;
      LITE_PARALLEL_COMMON_BEGIN(c, tid, c_round_down, 0, OUT_C_BLOCK) {
#ifdef LITE_USE_THREAD_POOL
        float* pre_out = pre_din + pre_in_size + tid * pre_out_size;
#elif defined(ARM_WITH_OMP)
        float* pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float* pre_out = pre_din + pre_in_size;
#endif
        const float* block_inr0 = cblock_inr0;
        const float* block_inr1 = cblock_inr1;
        const float* block_inr2 = cblock_inr2;
        const float* block_inr3 = cblock_inr3;
        const float* block_inr4 = cblock_inr4;

        const float* weight_c = weights + c * w_stride;
        const float* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = bias + c;
        }

#ifdef __aarch64__
        for (int hk = 0; hk < h_kernel; hk += OUT_H_BLOCK) {
          const float* wc0 = weight_c;

          const float* inr0 = block_inr0;
          const float* inr1 = block_inr1;
          const float* inr2 = block_inr2;
          const float* inr3 = block_inr3;
          const float* inr4 = block_inr4;

          float* pre_out0 = pre_out + hk * out_row_stride;
          float* pre_out1 = pre_out0 + out_row_stride;
          int cnt = w_loop;
          const float* r0 = inr0;
          const float* r1 = inr1;
          const float* r2 = inr2;
          const float* r3 = inr3;
          const float* r4 = inr4;
          const float* wc00 = wc0;
          float* ptr_out0 = pre_out0;
          float* ptr_out1 = pre_out1;

          // clang-format off
          asm volatile(
            "ldp    q0, q1,   [%[r0]], #32\n"
            "ldp    q9, q10,   [%[wc]], #32\n"
            "ldp    q19, q20,   [%[r2]], #32\n"
            "ldp    q2, q3,   [%[r0]], #32\n"
            "ldp    q21, q22,   [%[r2]], #32\n"
            "ldp    q11, q12,   [%[wc]], #32\n"
            "ldp    q4, q5,   [%[r0]], #32\n"
            "ldp    q23, q24,   [%[r2]], #32\n"
            "1:      \n"
            /* line 0 */
            // compute zero i00-i20
            "ldr    d6, [%[r0]] \n"
            "ldr    x20, [%[r0], #8]\n"
            "fmul   v15.4s,  v9.4s,  v0.s[0]\n"
            "ins   v6.d[1], x20\n"
            "ldr   d7, [%[r0], #16]\n"
            "fmul   v16.4s,  v9.4s,  v2.s[0]\n"
            "ldr   d25, [%[r2]]\n"
            "fmul   v17.4s,  v9.4s,  v4.s[0]\n"
            "ldr    x20, [%[r2], #8]\n"
            "fmul   v18.4s,  v9.4s,  v6.s[0]\n"
            "ins   v25.d[1], x20\n"
            "ldr    x20, [%[r0], #24]\n"
            "fmul   v28.4s,  v9.4s,  v19.s[0]\n"
            "fmul   v29.4s,  v9.4s,  v21.s[0]\n"
            "ldr   d26, [%[r2], #16]\n"
            "fmul   v30.4s,  v9.4s,  v23.s[0]\n"
            "ins   v7.d[1], x20\n"
            "ldr    x20, [%[r2], #24]\n"
            "fmul   v31.4s,  v9.4s,  v25.s[0]\n"
            "fmla   v15.4s,  v10.4s,  v0.s[1]\n"
            "ins   v26.d[1], x20\n"
            "ldr   d8, [%[r0], #32]\n"

            "fmla   v16.4s,  v10.4s,  v2.s[1]\n"
            "ldr    x20, [%[r0], #40]\n"
            "fmla   v17.4s,  v10.4s,  v4.s[1]\n"
            "ldr    d27, [%[r2], #32]\n"
            "ins    v8.d[1], x20\n"
            "fmla   v18.4s,  v10.4s,  v6.s[1]\n"
            "fmla   v28.4s,  v10.4s,  v19.s[1]\n"
            "ldr    x20, [%[r2], #40]\n"
            "fmla   v29.4s,  v10.4s,  v21.s[1]\n"
            "ldr    d13, [%[wc]]\n"
            "ins    v27.d[1], x20\n"
            "fmla   v30.4s,  v10.4s,  v23.s[1]\n"
            "fmla   v31.4s,  v10.4s,  v25.s[1]\n"
            "ldr    x20, [%[wc], #8]\n"
            "fmla   v15.4s,  v11.4s,  v0.s[2]\n"
            "ldr    d14, [%[wc], #16]\n"
            "ins    v13.d[1], x20\n"
            "fmla   v16.4s,  v11.4s,  v2.s[2]\n"
            "fmla   v17.4s,  v11.4s,  v4.s[2]\n"
            "ldr    x20, [%[wc], #24]\n"
            "fmla   v18.4s,  v11.4s,  v6.s[2]\n"
            "fmla   v28.4s,  v11.4s,  v19.s[2]\n"
            "ldr    d9, [%[wc], #32]\n"
            "ins    v14.d[1], x20\n"
            "fmla   v29.4s,  v11.4s,  v21.s[2]\n"
            "fmla   v30.4s,  v11.4s,  v23.s[2]\n"
            "ldr   x20, [%[wc], #40]\n"
            "fmla   v31.4s,  v11.4s,  v25.s[2]\n"
            "ldr   d10, [%[wc], #48]\n"
            "ins   v9.d[1], x20\n"
            // compute one i01-i21
            "fmla   v15.4s,  v12.4s,  v1.s[0]\n"
            "fmla   v16.4s,  v12.4s,  v3.s[0]\n"
            "ldr   x20, [%[wc], #56]\n"
            "fmla   v17.4s,  v12.4s,  v5.s[0]\n"
            "ldr   d11, [%[wc], #64]\n"
            "ins   v10.d[1], x20\n"
            "fmla   v18.4s,  v12.4s,  v7.s[0]\n"
            "ldr   x20, [%[wc], #72]\n"
            "fmla   v28.4s,  v12.4s,  v20.s[0]\n"
            "ldr   d0, [%[r1]]\n"
            "ins   v11.d[1], x20\n"
            "fmla   v29.4s,  v12.4s,  v22.s[0]\n"
            "fmla   v30.4s,  v12.4s,  v24.s[0]\n"
            "ldr   x20, [%[r1], #8]\n"
            "fmla   v31.4s,  v12.4s,  v26.s[0]\n"

            "fmla   v15.4s,  v13.4s,  v1.s[1]\n"
            "ldr   d12, [%[wc], 80]\n"
            "ins    v0.d[1], x20\n"
            "fmla   v16.4s,  v13.4s,  v3.s[1]\n"
            "ldr   x20, [%[wc], #88]\n"
            "fmla   v17.4s,  v13.4s,  v5.s[1]\n"
            "fmla   v18.4s,  v13.4s,  v7.s[1]\n"
            "ins   v12.d[1], x20\n"
            
            "fmla   v28.4s,  v13.4s,  v20.s[1]\n"
            "nop\n"
            "fmla   v29.4s,  v13.4s,  v22.s[1]\n"
            "fmla   v30.4s,  v13.4s,  v24.s[1]\n"
            "fmla   v31.4s,  v13.4s,  v26.s[1]\n"

            "fmla   v15.4s,  v14.4s,  v1.s[2]\n"
            "ldr    d1, [%[r1], #16]\n"
            "fmla   v16.4s,  v14.4s,  v3.s[2]\n"
            "ldr   x20, [%[r1], #24]\n"
            "fmla   v17.4s,  v14.4s,  v5.s[2]\n"
            "ldr    d13, [%[wc], #96]\n"
            "ins    v1.d[1], x20\n"
            "fmla   v18.4s,  v14.4s,  v7.s[2]\n"
            "ldr    x20, [%[wc], #104]\n"
            "fmla   v28.4s,  v14.4s,  v20.s[2]\n"
            "fmla   v29.4s,  v14.4s,  v22.s[2]\n"
            "ins    v13.d[1], x20\n"
            "fmla   v30.4s,  v14.4s,  v24.s[2]\n"
            "fmla   v31.4s,  v14.4s,  v26.s[2]\n"
            "ldr    d14, [%[wc], #112]\n"
            // compute two i02-i22
            "fmla   v15.4s,  v9.4s,  v2.s[0]\n"
            "fmla   v16.4s,  v9.4s,  v4.s[0]\n"
            "ldr    x20, [%[wc], #120]\n"
            "fmla   v17.4s,  v9.4s,  v6.s[0]\n"
            "fmla   v18.4s,  v9.4s,  v8.s[0]\n"
            "ins    v14.d[1], x20\n"
            "fmla   v28.4s,  v9.4s,  v21.s[0]\n"
            "fmla   v29.4s,  v9.4s,  v23.s[0]\n"
            "nop\n"
            "fmla   v30.4s,  v9.4s,  v25.s[0]\n"
            "fmla   v31.4s,  v9.4s,  v27.s[0]\n"
            "ldr    d9, [%[wc], #128]\n"

            "fmla   v15.4s,  v10.4s,  v2.s[1]\n"
            "fmla   v16.4s,  v10.4s,  v4.s[1]\n"
            "ldr    x20, [%[wc], #136]\n"
            "fmla   v17.4s,  v10.4s,  v6.s[1]\n"
            "fmla   v18.4s,  v10.4s,  v8.s[1]\n"
            "ins   v9.d[1], x20\n"
            "ldr    d3, [%[r1], #48]\n"
            "fmla   v28.4s,  v10.4s,  v21.s[1]\n"
            "fmla   v29.4s,  v10.4s,  v23.s[1]\n"
            "ldr    x20, [%[r1], #56]\n"
            "fmla   v30.4s,  v10.4s,  v25.s[1]\n"
            "fmla   v31.4s,  v10.4s,  v27.s[1]\n"
            "ins    v3.d[1], x20\n"

            "fmla   v15.4s,  v11.4s,  v2.s[2]\n"
            "ldr    d2, [%[r1], #32]\n"
            "fmla   v16.4s,  v11.4s,  v4.s[2]\n"
            "ldr    x20, [%[r1], #40]\n"
            "fmla   v17.4s,  v11.4s,  v6.s[2]\n"
            "fmla   v18.4s,  v11.4s,  v8.s[2]\n"
            "ldr    d4, [%[r1], #64]\n"
            "ins    v2.d[1], x20\n"
            "fmla   v28.4s,  v11.4s,  v21.s[2]\n"
            "ldr    x20, [%[r1], #72]\n"
            "fmla   v29.4s,  v11.4s,  v23.s[2]\n"
            "ldr    d5, [%[r1], #80]\n"
            "ins    v4.d[1], x20\n"
            "fmla   v30.4s,  v11.4s,  v25.s[2]\n"
            "ldr    x20, [%[r1], #88]\n"
            "fmla   v31.4s,  v11.4s,  v27.s[2]\n"
            "ldr    d6, [%[r1], #96]\n"
            "ins    v5.d[1], x20\n"
            /* line 1 */
            "fmla   v15.4s,  v12.4s,  v0.s[0]\n"
            "ldr    x20, [%[r1], #104]\n"
            "fmla   v16.4s,  v12.4s,  v2.s[0]\n"
            "ins    v6.d[1], x20\n"
            "ldr    d7, [%[r1], #112]\n"
            "fmla   v17.4s,  v12.4s,  v4.s[0]\n"
            "ldr    x20, [%[r1], #120]\n"

            "fmla   v18.4s,  v12.4s,  v6.s[0]\n"
            "ldr    d10, [%[wc], #144]\n"
            "ins    v7.d[1], x20\n"
            "fmla   v15.4s,  v13.4s,  v0.s[1]\n"
            "ldr    x20, [%[wc], #152]\n"
            "fmla   v16.4s,  v13.4s,  v2.s[1]\n"
            "ldr    d11, [%[wc], #160]\n"
            "ins    v10.d[1], x20\n"
            "fmla   v17.4s,  v13.4s,  v4.s[1]\n"
            "fmla   v18.4s,  v13.4s,  v6.s[1]\n"
            "ldr    x20, [%[wc], #168]\n"
            "fmla   v15.4s,  v14.4s,  v0.s[2]\n"
            "fmla   v16.4s,  v14.4s,  v2.s[2]\n"
            "ins    v11.d[1], x20\n"
            "ldr    d8,   [%[r1], #128]\n"
            "fmla   v17.4s,  v14.4s,  v4.s[2]\n"
            "ldr    x20, [%[r1], #136]\n"
            "fmla   v18.4s,  v14.4s,  v6.s[2]\n"
            "add    %[r1], %[r1], #128\n"

            "fmla   v15.4s,  v9.4s,  v1.s[0]\n"
            "ldr    d12, [%[wc], #176]\n"
            "ins    v8.d[1], x20\n"
            "fmla   v16.4s,  v9.4s,  v3.s[0]\n"
            "ldr    x20, [%[wc], #184]\n"
            "fmla   v17.4s,  v9.4s,  v5.s[0]\n"
            "fmla   v18.4s,  v9.4s,  v7.s[0]\n"
            "ldr    d13, [%[wc], #192]\n"
            "ins    v12.d[1], x20\n"
            "fmla   v15.4s,  v10.4s,  v1.s[1]\n"
            "ldr    x20, [%[wc], #200]\n"
            "fmla   v16.4s,  v10.4s,  v3.s[1]\n"
            "fmla   v17.4s,  v10.4s,  v5.s[1]\n"
            "ldr    d0, [%[r3]]\n"
            "ins    v13.d[1], x20\n"
            "fmla   v18.4s,  v10.4s,  v7.s[1]\n"
            "ldr    x20, [%[r3], #8]\n"
            "fmla   v15.4s,  v11.4s,  v1.s[2]\n"
            "fmla   v16.4s,  v11.4s,  v3.s[2]\n"
            "ldr    d1, [%[r3], #16]\n"
            "ins    v0.d[1], x20\n"
            "fmla   v17.4s,  v11.4s,  v5.s[2]\n"
            "fmla   v18.4s,  v11.4s,  v7.s[2]\n"
            "ldr    x20, [%[r3], #24]\n"
            
            "fmla   v15.4s,  v12.4s,  v2.s[0]\n"
            "fmla   v16.4s,  v12.4s,  v4.s[0]\n"
            "ldr    d14, [%[wc], #208]\n"
            "ins    v1.d[1], x20\n"
            "fmla   v17.4s,  v12.4s,  v6.s[0]\n"
            "ldr    x20, [%[wc], #216]\n"
            "fmla   v18.4s,  v12.4s,  v8.s[0]\n"
            "fmla   v15.4s,  v13.4s,  v2.s[1]\n"
            "ins    v14.d[1], x20\n"
            "ldr    d3, [%[r3], #48]\n"
            "fmla   v16.4s,  v13.4s,  v4.s[1]\n"
            "ldr    x20, [%[r3], #56]\n"
            "fmla   v17.4s,  v13.4s,  v6.s[1]\n"
            "ins    v3.d[1], x20\n"
            "ldr    d5, [%[r3], #80]\n"
            "fmla   v18.4s,  v13.4s,  v8.s[1]\n"
            "ldr    x20, [%[r3], #88]\n"
            "fmla   v15.4s,  v14.4s,  v2.s[2]\n"
            "ins    v5.d[1], x20\n"
            "ldr    d2, [%[r3], #32]\n"
            "fmla   v16.4s,  v14.4s,  v4.s[2]\n"
            "ldr    x20, [%[r3], #40]\n"
            "fmla   v17.4s,  v14.4s,  v6.s[2]\n"
            "ins    v2.d[1], x20\n"
            "ldr    d7, [%[r3], #112]\n"
            "fmla   v18.4s,  v14.4s,  v8.s[2]\n"
            "ldr    x20, [%[r3], #120]\n"

            "fmla   v28.4s,  v9.4s, v1.s[0]\n"
            "fmla   v29.4s,  v9.4s, v3.s[0]\n"
            "ins    v7.d[1], x20\n"
            "ldr    d4, [%[r3], #64]\n"
            "fmla   v30.4s,  v9.4s, v5.s[0]\n"
            "ldr    x20, [%[r3], #72]\n"
            "fmla   v31.4s,  v9.4s, v7.s[0]\n"
            "fmla   v28.4s,  v10.4s, v1.s[1]\n"
            "ldr    d6, [%[r3], #96]\n"
            "ins    v4.d[1], x20\n"
            "fmla   v29.4s,  v10.4s, v3.s[1]\n"
            "ldr    x20, [%[r3], #104]\n"
            "fmla   v30.4s,  v10.4s, v5.s[1]\n"
            "fmla   v31.4s,  v10.4s, v7.s[1]\n"
            "ins    v6.d[1], x20\n"
            "ldr    d8, [%[r3], #128]\n"
            "fmla   v28.4s,  v11.4s, v1.s[2]\n"
            "ldr    x20, [%[r3], #136]\n"
            "fmla   v29.4s,  v11.4s, v3.s[2]\n"
            "fmla   v30.4s,  v11.4s, v5.s[2]\n"
            "ldr    d9, [%[wc], #80]\n"
            "ins    v8.d[1], x20\n"
            "fmla   v31.4s,  v11.4s, v7.s[2]\n"
            "ldr    x20, [%[wc], #88]\n"

            "fmla   v28.4s,  v12.4s, v2.s[0]\n"
            "ldr    d10, [%[wc], #96]\n"
            "ins    v9.d[1], x20\n"
            "fmla   v29.4s,  v12.4s, v4.s[0]\n"
            "ldr    x20, [%[wc], #104]\n"
            "add    %[r3], %[r3], #128\n"
            "fmla   v30.4s,  v12.4s, v6.s[0]\n"
            "ldr    d11, [%[wc], #112]\n"
            "ins    v10.d[1], x20\n"
            "fmla   v31.4s,  v12.4s, v8.s[0]\n"
            "ldr    x20, [%[wc], #120]\n"

            "fmla   v28.4s,  v13.4s, v2.s[1]\n"
            "ldr    d12, [%[wc], #128]\n"
            "ins    v11.d[1], x20\n"
            "fmla   v29.4s,  v13.4s, v4.s[1]\n"
            "ldr    x20, [%[wc], #136]\n"
            "fmla   v30.4s,  v13.4s, v6.s[1]\n"
            "fmla   v31.4s,  v13.4s, v8.s[1]\n"

            "ins    v12.d[1], x20\n"
            "fmla   v28.4s,  v14.4s, v2.s[2]\n"
            "ldr    d13, [%[wc], #224]\n"
            "fmla   v29.4s,  v14.4s, v4.s[2]\n"
            "fmla   v30.4s,  v14.4s, v6.s[2]\n"
            "ldr    x20, [%[wc], #232]\n"
            "fmla   v31.4s,  v14.4s, v8.s[2]\n"

            "fmla   v28.4s,  v9.4s, v0.s[0]\n"
            "ldr    d14,  [%[wc], #240]\n"
            "ins    v13.d[1], x20\n"
            "fmla   v29.4s,  v9.4s, v2.s[0]\n"
            "ldr    x20, [%[wc], #248]\n"
            "fmla   v30.4s,  v9.4s, v4.s[0]\n"
            "fmla   v31.4s,  v9.4s, v6.s[0]\n"
            "ins    v14.d[1], x20\n"
            "ldr    d1, [%[r4], #16]\n"
            // line 2 w0-w2
            "fmla   v28.4s,  v10.4s, v0.s[1]\n"
            "ldr    x20, [%[r4], #24]\n"
            "fmla   v29.4s,  v10.4s, v2.s[1]\n"
            "fmla   v30.4s,  v10.4s, v4.s[1]\n"
            "ins    v1.d[1], x20\n"
            "fmla   v31.4s,  v10.4s, v6.s[1]\n"
            "fmla   v28.4s,  v11.4s, v0.s[2]\n"
            "ldr    d0, [%[r4]]\n"
            "fmla   v29.4s,  v11.4s, v2.s[2]\n"
            "ldr    x20, [%[r4], #8]\n"
            "fmla   v30.4s,  v11.4s, v4.s[2]\n"
            "ins    v0.d[1], x20\n"
            "ldr    d2, [%[r4], #32]\n"
            "fmla   v31.4s,  v11.4s, v6.s[2]\n"
            "ldr    x20, [%[r4], #40]\n"
            /* line 2 */
            "fmla   v15.4s,  v13.4s,  v19.s[0]\n"
            "ldr    d4, [%[r4], #64]\n"
            "ins    v2.d[1], x20\n"
            "fmla   v16.4s,  v13.4s,  v21.s[0]\n"
            "ldr    x20, [%[r4], #72]\n"
            "fmla   v17.4s,  v13.4s,  v23.s[0]\n"
            "ldr    d6, [%[r4], #96]\n"
            "ins    v4.d[1], x20\n"
            "fmla   v18.4s,  v13.4s,  v25.s[0]\n"
            "ldr    x20, [%[r4], #104]\n"
            "fmla   v28.4s,  v13.4s,  v0.s[0]\n"
            "ldr    d3, [%[r4], #48]\n"
            "ins    v6.d[1], x20\n"
            "fmla   v29.4s,  v13.4s,  v2.s[0]\n"
            "ldr    x20, [%[r4], #56]\n"
            "fmla   v30.4s,  v13.4s,  v4.s[0]\n"
            "ldr    d5, [%[r4], #80]\n"
            "ins    v3.d[1], x20\n"
            "fmla   v31.4s,  v13.4s,  v6.s[0]\n"
            "ldr    x20, [%[r4], #88]\n"

            "fmla   v15.4s,  v14.4s,  v19.s[1]\n"
            "ins    v5.d[1], x20\n"
            "ldr    d7, [%[r4], #112]\n"
            "fmla   v16.4s,  v14.4s,  v21.s[1]\n"
            "ldr    x20, [%[r4], #120]\n"
            "fmla   v17.4s,  v14.4s,  v23.s[1]\n"
            "ldr    d8, [%[r4], #128]\n"
            "ins    v7.d[1], x20\n"
            "fmla   v18.4s,  v14.4s,  v25.s[1]\n"
            "ldr    x20, [%[r4], #136]\n"
            "fmla   v28.4s,  v14.4s,  v0.s[1]\n"
            "ldr    d9, [%[wc], #256]\n"
            "ins    v8.d[1], x20\n"
            "fmla   v29.4s,  v14.4s,  v2.s[1]\n"
            "ldr    x20, [%[wc], #264]\n"
            "fmla   v30.4s,  v14.4s,  v4.s[1]\n"
            "fmla   v31.4s,  v14.4s,  v6.s[1]\n"
            "ins    v9.d[1], x20\n"
            "ldr    d10, [%[wc], #272]\n"
            "add    %[r4], %[r4], #128\n"

            "fmla   v15.4s,  v9.4s,  v19.s[2]\n"
            "ldr    x20, [%[wc], #280]\n"
            "fmla   v16.4s,  v9.4s,  v21.s[2]\n"
            "fmla   v17.4s,  v9.4s,  v23.s[2]\n"
            "ins    v10.d[1], x20\n"
            "ldr    d11, [%[wc], #288]\n"
            "fmla   v18.4s,  v9.4s,  v25.s[2]\n"
            "ldr    x20, [%[wc], #296]\n"
            "fmla   v28.4s,  v9.4s,  v0.s[2]\n"
            "fmla   v29.4s,  v9.4s,  v2.s[2]\n"
            "ins    v11.d[1], x20\n"
            "ldr    d12, [%[wc], #304]\n"
            "fmla   v30.4s,  v9.4s,  v4.s[2]\n"
            "ldr    x20, [%[wc], #312]\n"
            "fmla   v31.4s,  v9.4s,  v6.s[2]\n"

            "fmla   v15.4s,  v10.4s,  v20.s[0]\n"
            "ins    v12.d[1], x20\n"
            "ldr    d13, [%[wc], #320]\n"
            "fmla   v16.4s,  v10.4s,  v22.s[0]\n"
            "ldr    x20, [%[wc], #328]\n"
            "fmla   v17.4s,  v10.4s,  v24.s[0]\n"
            "fmla   v18.4s,  v10.4s,  v26.s[0]\n"
            "ins    v13.d[1], x20\n"
            "ldr    d14, [%[wc], #336]\n"
            "fmla   v28.4s,  v10.4s,  v1.s[0]\n"
            "ldr    x20, [%[wc], #344]\n"
            "fmla   v29.4s,  v10.4s,  v3.s[0]\n"
            "fmla   v30.4s,  v10.4s,  v5.s[0]\n"
            "ins    v14.d[1], x20\n"
            "ldr    d9, [%[wc], #352]\n"
            "fmla   v31.4s,  v10.4s,  v7.s[0]\n"

            "fmla   v15.4s,  v11.4s,  v20.s[1]\n"
            "ldr    x20, [%[wc], #360]\n"
            "fmla   v16.4s,  v11.4s,  v22.s[1]\n"
            "fmla   v17.4s,  v11.4s,  v24.s[1]\n"
            "ins    v9.d[1], x20\n"
            "ldr    d0, [%[r0], #32]\n"
            "fmla   v18.4s,  v11.4s,  v26.s[1]\n"
            "ldr    x20, [%[r0], #40]\n"
            "fmla   v28.4s,  v11.4s,  v1.s[1]\n"
            "fmla   v29.4s,  v11.4s,  v3.s[1]\n"
            "ins    v0.d[1], x20\n"
            "ldr    d19, [%[r2], #32]\n"
            "fmla   v30.4s,  v11.4s,  v5.s[1]\n"
            "ldr    x20, [%[r2], #40]\n"
            "fmla   v31.4s,  v11.4s,  v7.s[1]\n"
            
            "fmla   v15.4s,  v12.4s,  v20.s[2]\n"
            "ins    v19.d[1], x20\n"
            "ldr    d20, [%[r2], #48]\n"
            "fmla   v16.4s,  v12.4s,  v22.s[2]\n"
            "ldr    x20, [%[r2], #56]\n"
            "fmla   v17.4s,  v12.4s,  v24.s[2]\n"
            "fmla   v18.4s,  v12.4s,  v26.s[2]\n"
            "ins    v20.d[1], x20\n"
            "fmla   v28.4s,  v12.4s,  v1.s[2]\n"
            "ldr    d1, [%[r0], #48]\n"
            "fmla   v29.4s,  v12.4s,  v3.s[2]\n"
            "fmla   v30.4s,  v12.4s,  v5.s[2]\n"
            "ldr    x20, [%[r0], #56]\n"
            "fmla   v31.4s,  v12.4s,  v7.s[2]\n"

            "fmla   v15.4s,  v13.4s,  v21.s[0]\n"
            "ins    v1.d[1], x20\n"
            "ldr    d10, [%[wc], #-48]\n"
            "fmla   v16.4s,  v13.4s,  v23.s[0]\n"
            "ldr    x20, [%[wc], #-40]\n"
            "fmla   v17.4s,  v13.4s,  v25.s[0]\n"
            "fmla   v18.4s,  v13.4s,  v27.s[0]\n"
            "ldr    d11, [%[wc], #-32]\n"
            "ins    v10.d[1], x20\n"
            "fmla   v28.4s,  v13.4s,  v2.s[0]\n"
            "fmla   v29.4s,  v13.4s,  v4.s[0]\n"
            "ldr    x20, [%[wc], #-24]\n"
            "fmla   v30.4s,  v13.4s,  v6.s[0]\n"
            "fmla   v31.4s,  v13.4s,  v8.s[0]\n"
            "ldr    d12, [%[wc], #-16]\n"
            "ins    v11.d[1], x20\n"

            "fmla   v15.4s,  v14.4s,  v21.s[1]\n"
            "fmla   v16.4s,  v14.4s,  v23.s[1]\n"
            "ldr    x20, [%[wc], #-8]\n"
            "fmla   v17.4s,  v14.4s,  v25.s[1]\n"
            "fmla   v18.4s,  v14.4s,  v27.s[1]\n"
            "ins    v12.d[1], x20\n"
            "ldr    d22, [%[r2], 80]\n"
            "fmla   v28.4s,  v14.4s,  v2.s[1]\n"
            "fmla   v29.4s,  v14.4s,  v4.s[1]\n"
            "ldr    x20, [%[r2], #88]\n"
            "fmla   v30.4s,  v14.4s,  v6.s[1]\n"
            "fmla   v31.4s,  v14.4s,  v8.s[1]\n"
            "ins    v22.d[1], x20\n"
            "fmla   v15.4s,  v9.4s,  v21.s[2]\n"
            "ldr    d21, [%[r2], #64]\n"
            "fmla   v16.4s,  v9.4s,  v23.s[2]\n"
            "fmla   v17.4s,  v9.4s,  v25.s[2]\n"
            "ldr    x20, [%[r2], #72]\n"
            "fmla   v18.4s,  v9.4s,  v27.s[2]\n"
            "fmla   v28.4s,  v9.4s,  v2.s[2]\n"
            "ins    v21.d[1], x20\n"
            "ldr    d23, [%[r2], #96]\n"
            "fmla   v29.4s,  v9.4s,  v4.s[2]\n"
            "ldr    x20, [%[r2], #104]\n"
            "fmla   v30.4s,  v9.4s,  v6.s[2]\n"
            "ins    v23.d[1], x20\n"
            "ldr    d2, [%[r0], #64]\n"
            "fmla   v31.4s,  v9.4s,  v8.s[2]\n"
            "ldr    x20, [%[r0], #72]\n"
            "ldr    q9, [%[wc], #-64]\n"
            "subs   %w[cnt], %w[cnt], #1\n"
            "ins    v2.d[1], x20\n"
            "ldr    q3, [%[r0], #80]\n"
            "stp    q15, q16,  [%[ptr_out0]], #32\n"
            "ldr    q4, [%[r0], #96]\n"
            "stp    q28, q29,  [%[ptr_out1]], #32\n"
            "ldr    q5, [%[r0], #112]\n"
            "add    %[r0], %[r0], #128\n"
            "stp    q17, q18,  [%[ptr_out0]], #32\n"
            "ldr    q24, [%[r2], #112]\n"
            "stp    q30, q31,  [%[ptr_out1]], #32\n"
            "add    %[r2], %[r2], #128\n"
            "bne    1b\n"
            : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1),
              [r2] "+r"(r2), [r3] "+r"(r3), [r4] "+r"(r4),
              [wc]"+r"(wc00), [ptr_out0] "+r"(ptr_out0), [ptr_out1] "+r"(ptr_out1)
            : 
            : "cc","memory","v0","v1","v2","v3","v4", "x20",
              "v5","v6","v7","v8","v9","v10","v11","v12","v13",
              "v14","v15","v16","v17","v18", "v19", "v20", "v21", "v22",
              "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
          // clang-format on
          block_inr0 = block_inr4;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }
#else   // not __aarch64__
#endif  // __aarch64__
        write_to_output_c4_fp32(pre_out,
                                dout_batch,
                                c,
                                c + OUT_C_BLOCK,
                                h,
                                h + h_kernel,
                                0,
                                wout_round,
                                oc,
                                oh,
                                ow,
                                false,
                                nullptr,
                                &act_param,
                                bias_ptr);
      }
      LITE_PARALLEL_COMMON_END();
    }
  }
}
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
