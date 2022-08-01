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
#ifdef __aarch64__
const int OUT_C_BLOCK = 8;
const int OUT_H_BLOCK = 2;
const int OUT_W_BLOCK = 8;
#else
const int OUT_C_BLOCK = 8;
const int OUT_H_BLOCK = 1;
const int OUT_W_BLOCK = 8;
#endif

size_t conv3x3s1_direct_workspace_size(const operators::ConvParam& param,
                                       ARMContext* ctx) {
  auto dim_in = param.x->dims();
  auto dim_out = param.output->dims();
  auto paddings = *param.paddings;
  int ow = dim_out[3];
  int oh = dim_out[2];
  int ic = dim_in[1];
  DIRECT_WORKSPACE_COMPUTE(ctx, 3, 1, ow, oh, ic, OUT_C_BLOCK, OUT_H_BLOCK)
  return sizeof(float16_t) * (pre_in_size + ctx->threads() * pre_out_size);
}

// clang-format off
#ifdef __aarch64__
#define COMPUT_INIT                     \
  float16_t* ptr_out0 = pre_out0;       \
  float16_t* ptr_out1 = pre_out1;       \
  float16x8_t w0 = vld1q_f16(wc0);      \
  float16x8_t w1 = vld1q_f16(wc0 + 8);  \
  float16x8_t w2 = vld1q_f16(wc0 + 16); \
  float16x8_t w3 = vld1q_f16(wc0 + 24); \
  float16x8_t w4 = vld1q_f16(wc0 + 32); \
  float16x8_t w5 = vld1q_f16(wc0 + 40); \
  float16x8_t w6 = vld1q_f16(wc0 + 48); \
  float16x8_t w7 = vld1q_f16(wc0 + 56); \
  float16x8_t w8 = vld1q_f16(wc0 + 64); \
  const float16_t* r0 = inr0;           \
  const float16_t* r1 = inr1;           \
  const float16_t* r2 = inr2;           \
  const float16_t* r3 = inr3;

#define INIT_FIRST                   \
  "ldr q0, [%[r0]], #16\n"           \
  "ldr q4, [%[r1]], #16\n"           \
  "ldr q1, [%[r0]]\n"                \
  "ldr q5, [%[r1]]\n"                \
  "2:\n"                             \
  "fmul v16.8h, %[w0].8h, v0.h[0]\n" \
  "fmul v17.8h, %[w0].8h, v0.h[1]\n" \
  "fmul v18.8h, %[w0].8h, v0.h[2]\n" \
  "fmul v19.8h, %[w0].8h, v0.h[3]\n" \
  "fmul v20.8h, %[w0].8h, v0.h[4]\n" \
  "fmul v21.8h, %[w0].8h, v0.h[5]\n" \
  "fmul v22.8h, %[w0].8h, v0.h[6]\n" \
  "fmul v23.8h, %[w0].8h, v0.h[7]\n" \
  "fmul v24.8h, %[w0].8h, v4.h[0]\n" \
  "fmul v25.8h, %[w0].8h, v4.h[1]\n" \
  "fmul v26.8h, %[w0].8h, v4.h[2]\n" \
  "fmul v27.8h, %[w0].8h, v4.h[3]\n" \
  "fmul v28.8h, %[w0].8h, v4.h[4]\n" \
  "fmul v29.8h, %[w0].8h, v4.h[5]\n" \
  "fmul v30.8h, %[w0].8h, v4.h[6]\n" \
  "fmul v31.8h, %[w0].8h, v4.h[7]\n"

#define INIT                          \
  "ldr q0, [%[r0]], #16\n"            \
  "ldr q4, [%[r1]], #16\n"            \
  "ldr q1, [%[r0]]\n"                 \
  "ldr q5, [%[r1]]\n"                 \
  "2:\n"                              \
  "ldp q16, q17, [%[ptr_out0]]\n"     \
  "ldp q18, q19, [%[ptr_out0], #32]\n"\
  "ldp q24, q25, [%[ptr_out1]]\n"     \
  "ldp q20, q21, [%[ptr_out0], #64]\n"\
  "fmla v16.8h, %[w0].8h, v0.h[0]\n"  \
  "ldp q22, q23, [%[ptr_out0], #96]\n"\
  "fmla v17.8h, %[w0].8h, v0.h[1]\n"  \
  "ldp q26, q27, [%[ptr_out1], #32]\n"\
  "fmla v18.8h, %[w0].8h, v0.h[2]\n"  \
  "ldp q28, q29, [%[ptr_out1], #64]\n"\
  "fmla v19.8h, %[w0].8h, v0.h[3]\n"  \
  "ldp q30, q31, [%[ptr_out1], #96]\n"\
  "fmla v20.8h, %[w0].8h, v0.h[4]\n"  \
  "fmla v21.8h, %[w0].8h, v0.h[5]\n"  \
  "fmla v22.8h, %[w0].8h, v0.h[6]\n"  \
  "fmla v23.8h, %[w0].8h, v0.h[7]\n"  \
  "fmla v24.8h, %[w0].8h, v4.h[0]\n"  \
  "fmla v25.8h, %[w0].8h, v4.h[1]\n"  \
  "fmla v26.8h, %[w0].8h, v4.h[2]\n"  \
  "fmla v27.8h, %[w0].8h, v4.h[3]\n"  \
  "fmla v28.8h, %[w0].8h, v4.h[4]\n"  \
  "fmla v29.8h, %[w0].8h, v4.h[5]\n"  \
  "fmla v30.8h, %[w0].8h, v4.h[6]\n"  \
  "fmla v31.8h, %[w0].8h, v4.h[7]\n"

#define COMPUTE                        \
  /* r0-1 */                           \
  "fmla v16.8h, %[w1].8h, v0.h[1]\n"   \
  "fmla v17.8h, %[w1].8h, v0.h[2]\n"   \
  "fmla v18.8h, %[w1].8h, v0.h[3]\n"   \
  "fmla v19.8h, %[w1].8h, v0.h[4]\n"   \
  "fmla v20.8h, %[w1].8h, v0.h[5]\n"   \
  "fmla v21.8h, %[w1].8h, v0.h[6]\n"   \
  "fmla v22.8h, %[w1].8h, v0.h[7]\n"   \
  "fmla v23.8h, %[w1].8h, v1.h[0]\n"   \
  /* r1-1 */                           \
  "fmla v24.8h, %[w1].8h, v4.h[1]\n"   \
  "fmla v25.8h, %[w1].8h, v4.h[2]\n"   \
  "fmla v26.8h, %[w1].8h, v4.h[3]\n"   \
  "fmla v27.8h, %[w1].8h, v4.h[4]\n"   \
  "fmla v28.8h, %[w1].8h, v4.h[5]\n"   \
  "fmla v29.8h, %[w1].8h, v4.h[6]\n"   \
  "fmla v30.8h, %[w1].8h, v4.h[7]\n"   \
  "fmla v31.8h, %[w1].8h, v5.h[0]\n"   \
  /* r0-2 */                           \
  "fmla v16.8h, %[w2].8h, v0.h[2]\n"   \
  "fmla v17.8h, %[w2].8h, v0.h[3]\n"   \
  "fmla v18.8h, %[w2].8h, v0.h[4]\n"   \
  "fmla v19.8h, %[w2].8h, v0.h[5]\n"   \
  "fmla v20.8h, %[w2].8h, v0.h[6]\n"   \
  "fmla v21.8h, %[w2].8h, v0.h[7]\n"   \
  "fmla v22.8h, %[w2].8h, v1.h[0]\n"   \
  "fmla v23.8h, %[w2].8h, v1.h[1]\n"   \
  /* r1-2 */                           \
  "ldr q0, [%[r2]], #16\n"             \
  "fmla v24.8h, %[w2].8h, v4.h[2]\n"   \
  "fmla v25.8h, %[w2].8h, v4.h[3]\n"   \
  "fmla v26.8h, %[w2].8h, v4.h[4]\n"   \
  "fmla v27.8h, %[w2].8h, v4.h[5]\n"   \
  "ldr q1, [%[r2]]\n"                  \
  "fmla v28.8h, %[w2].8h, v4.h[6]\n"   \
  "fmla v29.8h, %[w2].8h, v4.h[7]\n"   \
  "fmla v30.8h, %[w2].8h, v5.h[0]\n"   \
  "fmla v31.8h, %[w2].8h, v5.h[1]\n"   \
  /* r1-0 */                           \
  "fmla v16.8h, %[w3].8h, v4.h[0]\n"   \
  "fmla v17.8h, %[w3].8h, v4.h[1]\n"   \
  "fmla v18.8h, %[w3].8h, v4.h[2]\n"   \
  "fmla v19.8h, %[w3].8h, v4.h[3]\n"   \
  "fmla v20.8h, %[w3].8h, v4.h[4]\n"   \
  "fmla v21.8h, %[w3].8h, v4.h[5]\n"   \
  "fmla v22.8h, %[w3].8h, v4.h[6]\n"   \
  "fmla v23.8h, %[w3].8h, v4.h[7]\n"   \
  /* r1-1 */                           \
  "fmla v16.8h, %[w4].8h, v4.h[1]\n"   \
  "fmla v17.8h, %[w4].8h, v4.h[2]\n"   \
  "fmla v18.8h, %[w4].8h, v4.h[3]\n"   \
  "fmla v19.8h, %[w4].8h, v4.h[4]\n"   \
  "fmla v20.8h, %[w4].8h, v4.h[5]\n"   \
  "fmla v21.8h, %[w4].8h, v4.h[6]\n"   \
  "fmla v22.8h, %[w4].8h, v4.h[7]\n"   \
  "fmla v23.8h, %[w4].8h, v5.h[0]\n"   \
  /* r1-2 */                           \
  "fmla v16.8h, %[w5].8h, v4.h[2]\n"   \
  "fmla v17.8h, %[w5].8h, v4.h[3]\n"   \
  "fmla v18.8h, %[w5].8h, v4.h[4]\n"   \
  "fmla v19.8h, %[w5].8h, v4.h[5]\n"   \
  "fmla v20.8h, %[w5].8h, v4.h[6]\n"   \
  "fmla v21.8h, %[w5].8h, v4.h[7]\n"   \
  "fmla v22.8h, %[w5].8h, v5.h[0]\n"   \
  "fmla v23.8h, %[w5].8h, v5.h[1]\n"   \
  "ldr q4, [%[r3]], #16\n"             \
  /* r2-0 */                           \
  "fmla v16.8h, %[w6].8h, v0.h[0]\n"   \
  "fmla v17.8h, %[w6].8h, v0.h[1]\n"   \
  "fmla v18.8h, %[w6].8h, v0.h[2]\n"   \
  "fmla v19.8h, %[w6].8h, v0.h[3]\n"   \
  "fmla v20.8h, %[w6].8h, v0.h[4]\n"   \
  "fmla v21.8h, %[w6].8h, v0.h[5]\n"   \
  "fmla v22.8h, %[w6].8h, v0.h[6]\n"   \
  "fmla v23.8h, %[w6].8h, v0.h[7]\n"   \
  "ldr q5, [%[r3]]\n"                  \
  "fmla v24.8h, %[w3].8h, v0.h[0]\n"   \
  "fmla v25.8h, %[w3].8h, v0.h[1]\n"   \
  "fmla v26.8h, %[w3].8h, v0.h[2]\n"   \
  "fmla v27.8h, %[w3].8h, v0.h[3]\n"   \
  "fmla v28.8h, %[w3].8h, v0.h[4]\n"   \
  "fmla v29.8h, %[w3].8h, v0.h[5]\n"   \
  "fmla v30.8h, %[w3].8h, v0.h[6]\n"   \
  "fmla v31.8h, %[w3].8h, v0.h[7]\n"   \
  /* r2-1 */                           \
  "fmla v16.8h, %[w7].8h, v0.h[1]\n"   \
  "fmla v17.8h, %[w7].8h, v0.h[2]\n"   \
  "fmla v18.8h, %[w7].8h, v0.h[3]\n"   \
  "fmla v19.8h, %[w7].8h, v0.h[4]\n"   \
  "fmla v20.8h, %[w7].8h, v0.h[5]\n"   \
  "fmla v21.8h, %[w7].8h, v0.h[6]\n"   \
  "fmla v22.8h, %[w7].8h, v0.h[7]\n"   \
  "fmla v23.8h, %[w7].8h, v1.h[0]\n"   \
  "fmla v24.8h, %[w4].8h, v0.h[1]\n"   \
  "fmla v25.8h, %[w4].8h, v0.h[2]\n"   \
  "fmla v26.8h, %[w4].8h, v0.h[3]\n"   \
  "fmla v27.8h, %[w4].8h, v0.h[4]\n"   \
  "fmla v28.8h, %[w4].8h, v0.h[5]\n"   \
  "fmla v29.8h, %[w4].8h, v0.h[6]\n"   \
  "fmla v30.8h, %[w4].8h, v0.h[7]\n"   \
  "fmla v31.8h, %[w4].8h, v1.h[0]\n"   \
  /* r2-2 */                           \
  "fmla v16.8h, %[w8].8h, v0.h[2]\n"   \
  "fmla v17.8h, %[w8].8h, v0.h[3]\n"   \
  "fmla v18.8h, %[w8].8h, v0.h[4]\n"   \
  "fmla v19.8h, %[w8].8h, v0.h[5]\n"   \
  "fmla v20.8h, %[w8].8h, v0.h[6]\n"   \
  "fmla v21.8h, %[w8].8h, v0.h[7]\n"   \
  "fmla v22.8h, %[w8].8h, v1.h[0]\n"   \
  "fmla v23.8h, %[w8].8h, v1.h[1]\n"   \
  "fmla v24.8h, %[w5].8h, v0.h[2]\n"   \
  "fmla v25.8h, %[w5].8h, v0.h[3]\n"   \
  "fmla v26.8h, %[w5].8h, v0.h[4]\n"   \
  "fmla v27.8h, %[w5].8h, v0.h[5]\n"   \
  "fmla v28.8h, %[w5].8h, v0.h[6]\n"   \
  "fmla v29.8h, %[w5].8h, v0.h[7]\n"   \
  "fmla v30.8h, %[w5].8h, v1.h[0]\n"   \
  "fmla v31.8h, %[w5].8h, v1.h[1]\n"   \
  /* r3-0 */                           \
  "fmla v24.8h, %[w6].8h, v4.h[0]\n"   \
  "fmla v25.8h, %[w6].8h, v4.h[1]\n"   \
  "fmla v26.8h, %[w6].8h, v4.h[2]\n"   \
  "fmla v27.8h, %[w6].8h, v4.h[3]\n"   \
  "fmla v28.8h, %[w6].8h, v4.h[4]\n"   \
  "fmla v29.8h, %[w6].8h, v4.h[5]\n"   \
  "fmla v30.8h, %[w6].8h, v4.h[6]\n"   \
  "fmla v31.8h, %[w6].8h, v4.h[7]\n"   \
  /* r3-1 */                           \
  "fmla v24.8h, %[w7].8h, v4.h[1]\n"   \
  "fmla v25.8h, %[w7].8h, v4.h[2]\n"   \
  "fmla v26.8h, %[w7].8h, v4.h[3]\n"   \
  "fmla v27.8h, %[w7].8h, v4.h[4]\n"   \
  "fmla v28.8h, %[w7].8h, v4.h[5]\n"   \
  "fmla v29.8h, %[w7].8h, v4.h[6]\n"   \
  "fmla v30.8h, %[w7].8h, v4.h[7]\n"   \
  "fmla v31.8h, %[w7].8h, v5.h[0]\n"   \
  /* r3-2 */                           \
  "fmla v24.8h, %[w8].8h, v4.h[2]\n"   \
  "fmla v25.8h, %[w8].8h, v4.h[3]\n"   \
  "stp q16, q17, [%[ptr_out0]], #32\n" \
  "fmla v26.8h, %[w8].8h, v4.h[4]\n"   \
  "fmla v27.8h, %[w8].8h, v4.h[5]\n"   \
  "stp q18, q19, [%[ptr_out0]], #32\n" \
  "fmla v28.8h, %[w8].8h, v4.h[6]\n"   \
  "fmla v29.8h, %[w8].8h, v4.h[7]\n"   \
  "stp q20, q21, [%[ptr_out0]], #32\n" \
  "fmla v30.8h, %[w8].8h, v5.h[0]\n"   \
  "fmla v31.8h, %[w8].8h, v5.h[1]\n"   \
  "stp q22, q23, [%[ptr_out0]], #32\n" \
  "subs   %w[cnt], %w[cnt], #1\n"      \
  "ldr q0, [%[r0]], #16\n"             \
  "stp q24, q25, [%[ptr_out1]], #32\n" \
  "ldr q4, [%[r1]], #16\n"             \
  "stp q26, q27, [%[ptr_out1]], #32\n" \
  "ldr q1, [%[r0]]\n"                  \
  "stp q28, q29, [%[ptr_out1]], #32\n" \
  "ldr q5, [%[r1]]\n"                  \
  "stp q30, q31, [%[ptr_out1]], #32\n" \
  "bne    2b\n"
#define ASM_PARAM                                  \
  : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), \
    [r2] "+r"(r2), [r3] "+r"(r3),                  \
    [ptr_out0] "+r"(ptr_out0),                     \
    [ptr_out1] "+r"(ptr_out1)                      \
  : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),      \
    [w3] "w"(w3), [w4] "w"(w4), [w5] "w"(w5),      \
    [w6] "w"(w6), [w7] "w"(w7), [w8] "w"(w8)       \
  : "cc", "memory", "v0", "v1", "v4", "v5", "v16", \
    "v17", "v17", "v18", "v19", "v20", "v21", "v22", \
    "v23", "v24", "v25", "v26", "v27", "v28", "v29", \
    "v30", "v31"
#else
#define COMPUT_INIT                     \
  float16_t* ptr_out0 = pre_out0;       \
  float16x8_t w0 = vld1q_f16(wc0);      \
  float16x8_t w1 = vld1q_f16(wc0 + 8);  \
  float16x8_t w2 = vld1q_f16(wc0 + 16); \
  const float16_t* r0 = inr0;           \
  const float16_t* r1 = inr1;           \
  const float16_t* r2 = inr2;

#define INIT_FIRST                   \
  "2:\n"                             \
  "vld1.32  {d6},  [%[r0]]!\n"       \
  "vldr d10, [%[wc], #0x30]\n"       \
  "vldr d11, [%[wc], #0x38]\n"       \
  "vldr d12, [%[wc], #0x40]\n"       \
  "vldr d13, [%[wc], #0x48]\n"       \
  "vld1.32  {d7},  [%[r0]]!\n"       \
  "vmul.f16 q8,  %q[w0],  d6[0]\n"   \
  "vmul.f16 q9,  %q[w0],  d6[1]\n"   \
  "vmul.f16 q10, %q[w0],  d6[2]\n"   \
  "vmul.f16 q11, %q[w0],  d6[3]\n"   \
  "vld1.32 {d4}, [%[r0]]\n"          \
  "vmul.f16 q12, %q[w0],  d7[0]\n"   \
  "vmul.f16 q13, %q[w0],  d7[1]\n"   \
  "vmul.f16 q14, %q[w0],  d7[2]\n"   \
  "vmul.f16 q15, %q[w0],  d7[3]\n"

#define INIT                           \
  "2:\n"                               \
  "vld1.32 {d16-d19}, [%[ptr_out0]]!\n"\
  "vld1.32  {d6},  [%[r0]]!\n"         \
  "vldr d10, [%[wc], #0x30]\n"         \
  "vldr d11, [%[wc], #0x38]\n"         \
  "vld1.32 {d20-d23}, [%[ptr_out0]]!\n"\
  "vldr d12, [%[wc], #0x40]\n"         \
  "vldr d13, [%[wc], #0x48]\n"         \
  "vld1.32  {d7},  [%[r0]]!\n"         \
  "vld1.32 {d24-d27}, [%[ptr_out0]]!\n"\
  "vmla.f16 q8,  %q[w0], d6[0]\n"      \
  "vmla.f16 q9,  %q[w0], d6[1]\n"      \
  "vld1.32 {d28-d31}, [%[ptr_out0]]\n" \
  "vmla.f16 q10, %q[w0], d6[2]\n"      \
  "vmla.f16 q11, %q[w0], d6[3]\n"      \
  "vld1.32 {d4}, [%[r0]]\n"            \
  "sub      %[ptr_out0], #96\n"        \
  "vmla.f16 q12, %q[w0], d7[0]\n"      \
  "vmla.f16 q13, %q[w0], d7[1]\n"      \
  "vmla.f16 q14, %q[w0], d7[2]\n"      \
  "vmla.f16 q15, %q[w0], d7[3]\n"

#define COMPUTE                        \
  /* r0-1 */                           \
  "vmla.f16 q8,  %q[w1], d6[1]\n"      \
  "vmla.f16 q9,  %q[w1], d6[2]\n"      \
  "vmla.f16 q10, %q[w1], d6[3]\n"      \
  "vmla.f16 q11, %q[w1], d7[0]\n"      \
  "vmla.f16 q12, %q[w1], d7[1]\n"      \
  "vmla.f16 q13, %q[w1], d7[2]\n"      \
  "vmla.f16 q14, %q[w1], d7[3]\n"      \
  "vmla.f16 q15, %q[w1], d4[0]\n"      \
  /* r0-2 */                           \
  "vmla.f16 q8,  %q[w2], d6[2]\n"      \
  "vmla.f16 q9,  %q[w2], d6[3]\n"      \
  "vld1.32  {d6},  [%[r1]]!\n"         \
  "vmla.f16 q10, %q[w2], d7[0]\n"      \
  "vmla.f16 q11, %q[w2], d7[1]\n"      \
  "vldr d14, [%[wc], #0x50]\n"         \
  "vmla.f16 q12, %q[w2], d7[2]\n"      \
  "vmla.f16 q13, %q[w2], d7[3]\n"      \
  "vld1.32  {d7},  [%[r1]]!\n"         \
  "vmla.f16 q14, %q[w2], d4[0]\n"      \
  "vmla.f16 q15, %q[w2], d4[1]\n"      \
  "vldr d15, [%[wc], #0x58]\n"         \
  "vldr    d4,  [%[r1]]\n"             \
  /* r1-0 */                           \
  "vmla.f16 q8,  q5, d6[0]\n"          \
  "vmla.f16 q9,  q5, d6[1]\n"          \
  "vmla.f16 q10, q5, d6[2]\n"          \
  "vmla.f16 q11, q5, d6[3]\n"          \
  "vmla.f16 q12, q5, d7[0]\n"          \
  "vmla.f16 q13, q5, d7[1]\n"          \
  "vmla.f16 q14, q5, d7[2]\n"          \
  "vmla.f16 q15, q5, d7[3]\n"          \
  "vldr d10, [%[wc], #0x60]\n"         \
  /* r1-1 */                           \
  "vmla.f16 q8,  q6, d6[1]\n"          \
  "vmla.f16 q9,  q6, d6[2]\n"          \
  "vmla.f16 q10, q6, d6[3]\n"          \
  "vmla.f16 q11, q6, d7[0]\n"          \
  "vldr d11, [%[wc], #0x68]\n"         \
  "vmla.f16 q12, q6, d7[1]\n"          \
  "vmla.f16 q13, q6, d7[2]\n"          \
  "vmla.f16 q14, q6, d7[3]\n"          \
  "vmla.f16 q15, q6, d4[0]\n"          \
  "vldr d12, [%[wc], #0x70]\n"         \
  /* r1-2 */                           \
  "vmla.f16 q8,  q7, d6[2]\n"          \
  "vmla.f16 q9,  q7, d6[3]\n"          \
  "vld1.32  {d6},  [%[r2]]!\n"         \
  "vmla.f16 q10, q7, d7[0]\n"          \
  "vmla.f16 q11, q7, d7[1]\n"          \
  "vldr d13, [%[wc], #0x78]\n"         \
  "vmla.f16 q12, q7, d7[2]\n"          \
  "vmla.f16 q13, q7, d7[3]\n"          \
  "vld1.32  {d7},  [%[r2]]!\n"         \
  "vmla.f16 q14, q7, d4[0]\n"          \
  "vmla.f16 q15, q7, d4[1]\n"          \
  "vldr d14, [%[wc], #0x80]\n"         \
  "vldr    d4,  [%[r2]]\n"             \
  "vldr d15, [%[wc], #0x88]\n"         \
  /* r2-0 */                           \
  "vmla.f16 q8,  q5, d6[0]\n"          \
  "vmla.f16 q9,  q5, d6[1]\n"          \
  "vmla.f16 q10, q5, d6[2]\n"          \
  "vmla.f16 q11, q5, d6[3]\n"          \
  "vmla.f16 q12, q5, d7[0]\n"          \
  "vmla.f16 q13, q5, d7[1]\n"          \
  "vmla.f16 q14, q5, d7[2]\n"          \
  "vmla.f16 q15, q5, d7[3]\n"          \
  /* r2-1 */                           \
  "vmla.f16 q8,  q6, d6[1]\n"          \
  "vmla.f16 q9,  q6, d6[2]\n"          \
  "vmla.f16 q10, q6, d6[3]\n"          \
  "vmla.f16 q11, q6, d7[0]\n"          \
  "vmla.f16 q12, q6, d7[1]\n"          \
  "vmla.f16 q13, q6, d7[2]\n"          \
  "vmla.f16 q14, q6, d7[3]\n"          \
  "vmla.f16 q15, q6, d4[0]\n"          \
  /* r2-2 */                           \
  "vmla.f16 q8,  q7, d6[2]\n"          \
  "vmla.f16 q9,  q7, d6[3]\n"          \
  "vmla.f16 q10, q7, d7[0]\n"          \
  "vmla.f16 q11, q7, d7[1]\n"          \
  "vmla.f16 q12, q7, d7[2]\n"          \
  "vmla.f16 q13, q7, d7[3]\n"          \
  "vst1.32  {d16-d19}, [%[ptr_out0]]!\n"\
  "subs %[cnt], #1\n"                  \
  "vmla.f16 q14, q7, d4[0]\n"          \
  "vmla.f16 q15, q7, d4[1]\n"          \
  "vst1.32  {d20-d23}, [%[ptr_out0]]!\n"\
  "vst1.32  {d24-d27}, [%[ptr_out0]]!\n"\
  "vst1.32  {d28-d31}, [%[ptr_out0]]!\n"\
  "bne    2b\n"
#define ASM_PARAM                                  \
  : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), \
    [r2] "+r"(r2), [ptr_out0] "+r"(ptr_out0)       \
  : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),      \
    [wc] "r"(wc0)                                  \
  : "cc", "memory", "q2", "q3", "q5", "q6", "q7",  \
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", \
    "q15"

#endif
// clang-format on

void conv_3x3s1_direct_fp16(const float16_t* i_data,
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
  auto act_param = param.activation_param;
  const int pad_w = paddings[2];
  const int pad_h = paddings[0];
  DIRECT_WORKSPACE_COMPUTE(ctx, 3, 1, ow, oh, ic, OUT_C_BLOCK, OUT_H_BLOCK)

  float16_t* tmp_work_space = ctx->workspace_data<float16_t>();
  float16_t ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(float16_t) * win_round);

  //! l2_cache start
  float16_t* pre_din = tmp_work_space;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;
  int w_stride = ic * 9; /*kernel_w * kernel_h*/
  int w_stride_chin = OUT_C_BLOCK * 9;

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round >> 3;

  int c_remain = oc - (oc / OUT_C_BLOCK) * OUT_C_BLOCK;
  int c_round_down = (oc / OUT_C_BLOCK) * OUT_C_BLOCK;
  if (c_remain) {
    c_round_down++;
  }

  int out_row_stride = OUT_C_BLOCK * wout_round;
  auto act_type = act_param.active_type;
  bool flag_bias = param.bias != nullptr;
  float16_t alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3

  float16_t offset = 0.f;
  float16_t threshold = 6.f;

  if (act_param.has_active) {
    act_acquire(act_type, flag_act, alpha, offset, threshold, act_param);
  }

  for (int n = 0; n < bs; ++n) {
    const float16_t* din_batch = i_data + n * ic * size_in_channel;
    float16_t* dout_batch = o_data + n * oc * size_out_channel;
    for (int h = 0; h < oh; h += hout_r_block) {
      int h_kernel = hout_r_block;
      int hs = h - pad_h;
      int he = hs + h_kernel + 2;
      if (h + hout_r_block > oh) {
        h_kernel = oh - h;
      }
      prepack_input_nxw(
          din_batch, pre_din, 0, ic, hs, he, ws, we, ic, win, ih, ptr_zero);

      const float16_t* cblock_inr0 = pre_din;
      const float16_t* cblock_inr1 = cblock_inr0 + in_len;
      const float16_t* cblock_inr2 = cblock_inr1 + in_len;
      const float16_t* cblock_inr3 = cblock_inr2 + in_len;

      LITE_PARALLEL_COMMON_BEGIN(c, tid, c_round_down, 0, OUT_C_BLOCK) {
#ifdef LITE_USE_THREAD_POOL
        float16_t* pre_out = pre_din + pre_in_size + tid * pre_out_size;
#elif ARM_WITH_OMP
        float16_t* pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float16_t* pre_out = pre_din + pre_in_size;
#endif
        const float16_t* block_inr0 = cblock_inr0;
        const float16_t* block_inr1 = cblock_inr1;
        const float16_t* block_inr2 = cblock_inr2;
        const float16_t* block_inr3 = cblock_inr3;

        const float16_t* weight_c = weights + c * w_stride;
        const float16_t* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = bias + c;
        }

        for (int hk = 0; hk < h_kernel; hk += OUT_H_BLOCK) {
          const float16_t* wc0 = weight_c;
          const float16_t* inr0 = block_inr0;
          const float16_t* inr1 = block_inr1;
          const float16_t* inr2 = block_inr2;

          float16_t* pre_out0 = pre_out + hk * out_row_stride;
#ifdef __aarch64__
          const float16_t* inr3 = block_inr3;
          float16_t* pre_out1 = pre_out0 + out_row_stride;
          // first
          if (1) {
            COMPUT_INIT

            int cnt = w_loop;
            asm volatile(INIT_FIRST COMPUTE ASM_PARAM);
            wc0 += 9 * OUT_C_BLOCK;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
          }
          for (int i = 0; i < ic - 1; ++i) {
            COMPUT_INIT

            int cnt = w_loop;
            asm volatile(INIT COMPUTE ASM_PARAM);
            wc0 += 9 * OUT_C_BLOCK;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
          }
          block_inr0 = block_inr2;
          block_inr1 = block_inr3;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
#else   // not __aarch64__
          // first
          if (1) {
            COMPUT_INIT

            int cnt = w_loop;
            asm volatile(INIT_FIRST COMPUTE ASM_PARAM);
            wc0 += 9 * OUT_C_BLOCK;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
          }
          for (int i = 0; i < ic - 1; ++i) {
            COMPUT_INIT

            int cnt = w_loop;
            asm volatile(INIT COMPUTE ASM_PARAM);
            wc0 += 9 * OUT_C_BLOCK;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
          }
          block_inr0 = block_inr1;
          block_inr1 = block_inr2;
          block_inr2 = block_inr1 + in_len;
#endif  // __aarch64__
        }
        write_to_oc8_fp16(pre_out,
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
                          flag_act,
                          alpha,
                          bias_ptr,
                          flag_bias,
                          offset,
                          threshold);
      }
      LITE_PARALLEL_COMMON_END();
    }
  }
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
