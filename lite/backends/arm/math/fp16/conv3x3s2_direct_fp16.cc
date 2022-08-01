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

const int OUT_C_BLOCK = 8;
const int OUT_H_BLOCK = 2;
#ifdef __aarch64__
const int OUT_W_BLOCK = 8;
#else
const int OUT_W_BLOCK = 4;
#endif

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
  const float16_t* r3 = inr3;           \
  const float16_t* r4 = inr4;
#else
#define COMPUT_INIT               \
  float16_t* ptr_out0 = pre_out0; \
  float16_t* ptr_out1 = pre_out1; \
  const float16_t* r0 = inr0;     \
  const float16_t* r1 = inr1;     \
  const float16_t* r2 = inr2;     \
  const float16_t* r3 = inr3;     \
  const float16_t* r4 = inr4;
#endif

size_t conv3x3s2_direct_workspace_size(const operators::ConvParam& param,
                                       ARMContext* ctx) {
  auto dim_in = param.x->dims();
  auto dim_out = param.output->dims();
  auto paddings = *param.paddings;
  int ow = dim_out[3];
  int oh = dim_out[2];
  int ic = dim_in[1];
  DIRECT_WORKSPACE_COMPUTE(ctx, 3, 2, ow, oh, ic, OUT_C_BLOCK, OUT_H_BLOCK)
  return sizeof(float16_t) * (pre_in_size + ctx->threads() * pre_out_size);
}

// clang-format off
#ifdef __aarch64__
#define INIT_FIRST                   \
  "2:\n"                             \
  "ldp q0, q1, [%[r0]], #32\n"       \
  "ldp q4, q5, [%[r2]], #32\n"       \
  "ldr d10, [%[r0]]\n"               \
  "ldr d12, [%[r2]]\n"               \
  "fmul v16.8h, %[w0].8h, v0.h[0]\n" \
  "fmul v17.8h, %[w0].8h, v0.h[2]\n" \
  "fmul v18.8h, %[w0].8h, v0.h[4]\n" \
  "fmul v19.8h, %[w0].8h, v0.h[6]\n" \
  "fmul v20.8h, %[w0].8h, v1.h[0]\n" \
  "fmul v21.8h, %[w0].8h, v1.h[2]\n" \
  "fmul v22.8h, %[w0].8h, v1.h[4]\n" \
  "fmul v23.8h, %[w0].8h, v1.h[6]\n" \
  "fmul v24.8h, %[w0].8h, v4.h[0]\n" \
  "fmul v25.8h, %[w0].8h, v4.h[2]\n" \
  "fmul v26.8h, %[w0].8h, v4.h[4]\n" \
  "fmul v27.8h, %[w0].8h, v4.h[6]\n" \
  "fmul v28.8h, %[w0].8h, v5.h[0]\n" \
  "fmul v29.8h, %[w0].8h, v5.h[2]\n" \
  "fmul v30.8h, %[w0].8h, v5.h[4]\n" \
  "fmul v31.8h, %[w0].8h, v5.h[6]\n"

#define INIT                          \
  "2:\n"                              \
  "ldp q16, q17, [%[ptr_out0]]\n"     \
  "ldp q0, q1, [%[r0]], #32\n"        \
  "ldp q18, q19, [%[ptr_out0], #32]\n"\
  "ldp q4, q5, [%[r2]], #32\n"        \
  "ldp q20, q21, [%[ptr_out0], #64]\n"\
  "ldr d10, [%[r0]]\n"                \
  "ldp q22, q23, [%[ptr_out0], #96]\n"\
  "ldr d12, [%[r2]]\n"                \
  "ldp q24, q25, [%[ptr_out1]]\n"     \
  "fmla v16.8h, %[w0].8h, v0.h[0]\n"  \
  "ldp q26, q27, [%[ptr_out1], #32]\n"\
  "fmla v17.8h, %[w0].8h, v0.h[2]\n"  \
  "ldp q28, q29, [%[ptr_out1], #64]\n"\
  "fmla v18.8h, %[w0].8h, v0.h[4]\n"  \
  "ldp q30, q31, [%[ptr_out1], #96]\n"\
  "fmla v19.8h, %[w0].8h, v0.h[6]\n"  \
  "fmla v20.8h, %[w0].8h, v1.h[0]\n"  \
  "fmla v21.8h, %[w0].8h, v1.h[2]\n"  \
  "fmla v22.8h, %[w0].8h, v1.h[4]\n"  \
  "fmla v23.8h, %[w0].8h, v1.h[6]\n"  \
  "fmla v24.8h, %[w0].8h, v4.h[0]\n"  \
  "fmla v25.8h, %[w0].8h, v4.h[2]\n"  \
  "fmla v26.8h, %[w0].8h, v4.h[4]\n"  \
  "fmla v27.8h, %[w0].8h, v4.h[6]\n"  \
  "fmla v28.8h, %[w0].8h, v5.h[0]\n"  \
  "fmla v29.8h, %[w0].8h, v5.h[2]\n"  \
  "fmla v30.8h, %[w0].8h, v5.h[4]\n"  \
  "fmla v31.8h, %[w0].8h, v5.h[6]\n"

#define COMPUTE                        \
  /* r2-0 */                           \
  "fmla v16.8h, %[w6].8h, v4.h[0]\n"   \
  "fmla v17.8h, %[w6].8h, v4.h[2]\n"   \
  "fmla v18.8h, %[w6].8h, v4.h[4]\n"   \
  "fmla v19.8h, %[w6].8h, v4.h[6]\n"   \
  "fmla v20.8h, %[w6].8h, v5.h[0]\n"   \
  "fmla v21.8h, %[w6].8h, v5.h[2]\n"   \
  "fmla v22.8h, %[w6].8h, v5.h[4]\n"   \
  "fmla v23.8h, %[w6].8h, v5.h[6]\n"   \
  /* r0-1 */                           \
  "fmla v16.8h, %[w1].8h, v0.h[1]\n"   \
  "fmla v17.8h, %[w1].8h, v0.h[3]\n"   \
  "fmla v18.8h, %[w1].8h, v0.h[5]\n"   \
  "fmla v19.8h, %[w1].8h, v0.h[7]\n"   \
  "fmla v20.8h, %[w1].8h, v1.h[1]\n"   \
  "fmla v21.8h, %[w1].8h, v1.h[3]\n"   \
  "fmla v22.8h, %[w1].8h, v1.h[5]\n"   \
  "fmla v23.8h, %[w1].8h, v1.h[7]\n"   \
  /* r2-1 */                           \
  "fmla v24.8h, %[w1].8h, v4.h[1]\n"   \
  "fmla v25.8h, %[w1].8h, v4.h[3]\n"   \
  "fmla v26.8h, %[w1].8h, v4.h[5]\n"   \
  "fmla v27.8h, %[w1].8h, v4.h[7]\n"   \
  "fmla v28.8h, %[w1].8h, v5.h[1]\n"   \
  "fmla v29.8h, %[w1].8h, v5.h[3]\n"   \
  "fmla v30.8h, %[w1].8h, v5.h[5]\n"   \
  "fmla v31.8h, %[w1].8h, v5.h[7]\n"   \
  /* r2-1 */                           \
  "fmla v16.8h, %[w7].8h, v4.h[1]\n"   \
  "fmla v17.8h, %[w7].8h, v4.h[3]\n"   \
  "fmla v18.8h, %[w7].8h, v4.h[5]\n"   \
  "fmla v19.8h, %[w7].8h, v4.h[7]\n"   \
  "fmla v20.8h, %[w7].8h, v5.h[1]\n"   \
  "fmla v21.8h, %[w7].8h, v5.h[3]\n"   \
  "fmla v22.8h, %[w7].8h, v5.h[5]\n"   \
  "fmla v23.8h, %[w7].8h, v5.h[7]\n"   \
  /* r0-2 */                           \
  "fmla v16.8h, %[w2].8h, v0.h[2]\n"   \
  "fmla v17.8h, %[w2].8h, v0.h[4]\n"   \
  "fmla v18.8h, %[w2].8h, v0.h[6]\n"   \
  "fmla v19.8h, %[w2].8h, v1.h[0]\n"   \
  "fmla v20.8h, %[w2].8h, v1.h[2]\n"   \
  "fmla v21.8h, %[w2].8h, v1.h[4]\n"   \
  "fmla v22.8h, %[w2].8h, v1.h[6]\n"   \
  "fmla v23.8h, %[w2].8h, v10.h[0]\n"  \
  "ldp q0, q1, [%[r1]], #32\n"         \
  /* r2-2 */                           \
  "fmla v24.8h, %[w2].8h, v4.h[2]\n"   \
  "fmla v25.8h, %[w2].8h, v4.h[4]\n"   \
  "fmla v26.8h, %[w2].8h, v4.h[6]\n"   \
  "fmla v27.8h, %[w2].8h, v5.h[0]\n"   \
  "ldr d10, [%[r1]]\n"                 \
  "fmla v28.8h, %[w2].8h, v5.h[2]\n"   \
  "fmla v29.8h, %[w2].8h, v5.h[4]\n"   \
  "fmla v30.8h, %[w2].8h, v5.h[6]\n"   \
  "fmla v31.8h, %[w2].8h, v12.h[0]\n"  \
  "fmla v16.8h, %[w8].8h, v4.h[2]\n"   \
  "fmla v17.8h, %[w8].8h, v4.h[4]\n"   \
  "fmla v18.8h, %[w8].8h, v4.h[6]\n"   \
  "fmla v19.8h, %[w8].8h, v5.h[0]\n"   \
  "fmla v20.8h, %[w8].8h, v5.h[2]\n"   \
  "fmla v21.8h, %[w8].8h, v5.h[4]\n"   \
  "fmla v22.8h, %[w8].8h, v5.h[6]\n"   \
  "fmla v23.8h, %[w8].8h, v12.h[0]\n"  \
  "ldp q4, q5, [%[r3]], #32\n"         \
  /* r1-0 */                           \
  "fmla v16.8h, %[w3].8h, v0.h[0]\n"   \
  "fmla v17.8h, %[w3].8h, v0.h[2]\n"   \
  "fmla v18.8h, %[w3].8h, v0.h[4]\n"   \
  "fmla v19.8h, %[w3].8h, v0.h[6]\n"   \
  "ldr d12, [%[r3]]\n"                 \
  "fmla v20.8h, %[w3].8h, v1.h[0]\n"   \
  "fmla v21.8h, %[w3].8h, v1.h[2]\n"   \
  "fmla v22.8h, %[w3].8h, v1.h[4]\n"   \
  "fmla v23.8h, %[w3].8h, v1.h[6]\n"   \
  /* r3-0 */                           \
  "fmla v24.8h, %[w3].8h, v4.h[0]\n"   \
  "fmla v25.8h, %[w3].8h, v4.h[2]\n"   \
  "fmla v26.8h, %[w3].8h, v4.h[4]\n"   \
  "fmla v27.8h, %[w3].8h, v4.h[6]\n"   \
  "fmla v28.8h, %[w3].8h, v5.h[0]\n"   \
  "fmla v29.8h, %[w3].8h, v5.h[2]\n"   \
  "fmla v30.8h, %[w3].8h, v5.h[4]\n"   \
  "fmla v31.8h, %[w3].8h, v5.h[6]\n"   \
  /* r1-1 */                           \
  "fmla v16.8h, %[w4].8h, v0.h[1]\n"   \
  "fmla v17.8h, %[w4].8h, v0.h[3]\n"   \
  "fmla v18.8h, %[w4].8h, v0.h[5]\n"   \
  "fmla v19.8h, %[w4].8h, v0.h[7]\n"   \
  "fmla v20.8h, %[w4].8h, v1.h[1]\n"   \
  "fmla v21.8h, %[w4].8h, v1.h[3]\n"   \
  "fmla v22.8h, %[w4].8h, v1.h[5]\n"   \
  "fmla v23.8h, %[w4].8h, v1.h[7]\n"   \
  /* r3-1 */                           \
  "fmla v24.8h, %[w4].8h, v4.h[1]\n"   \
  "fmla v25.8h, %[w4].8h, v4.h[3]\n"   \
  "fmla v26.8h, %[w4].8h, v4.h[5]\n"   \
  "fmla v27.8h, %[w4].8h, v4.h[7]\n"   \
  "fmla v28.8h, %[w4].8h, v5.h[1]\n"   \
  "fmla v29.8h, %[w4].8h, v5.h[3]\n"   \
  "fmla v30.8h, %[w4].8h, v5.h[5]\n"   \
  "fmla v31.8h, %[w4].8h, v5.h[7]\n"   \
  /* r1-2 */                           \
  "fmla v16.8h, %[w5].8h, v0.h[2]\n"   \
  "fmla v17.8h, %[w5].8h, v0.h[4]\n"   \
  "fmla v18.8h, %[w5].8h, v0.h[6]\n"   \
  "fmla v19.8h, %[w5].8h, v1.h[0]\n"   \
  "fmla v20.8h, %[w5].8h, v1.h[2]\n"   \
  "fmla v21.8h, %[w5].8h, v1.h[4]\n"   \
  "fmla v22.8h, %[w5].8h, v1.h[6]\n"   \
  "fmla v23.8h, %[w5].8h, v10.h[0]\n"  \
  /* r3-2 */                           \
  "ldp q0, q1, [%[r4]], #32\n"         \
  "fmla v24.8h, %[w5].8h, v4.h[2]\n"   \
  "fmla v25.8h, %[w5].8h, v4.h[4]\n"   \
  "fmla v26.8h, %[w5].8h, v4.h[6]\n"   \
  "fmla v27.8h, %[w5].8h, v5.h[0]\n"   \
  "ldr d10, [%[r4]]\n"                 \
  "fmla v28.8h, %[w5].8h, v5.h[2]\n"   \
  "fmla v29.8h, %[w5].8h, v5.h[4]\n"   \
  "fmla v30.8h, %[w5].8h, v5.h[6]\n"   \
  "fmla v31.8h, %[w5].8h, v12.h[0]\n"  \
  /* r4-0 */                           \
  "fmla v24.8h, %[w6].8h, v0.h[0]\n"   \
  "fmla v25.8h, %[w6].8h, v0.h[2]\n"   \
  "fmla v26.8h, %[w6].8h, v0.h[4]\n"   \
  "fmla v27.8h, %[w6].8h, v0.h[6]\n"   \
  "stp q16, q17, [%[ptr_out0]], #32\n" \
  "fmla v28.8h, %[w6].8h, v1.h[0]\n"   \
  "stp q18, q19, [%[ptr_out0]], #32\n" \
  "fmla v29.8h, %[w6].8h, v1.h[2]\n"   \
  "stp q20, q21, [%[ptr_out0]], #32\n" \
  "fmla v30.8h, %[w6].8h, v1.h[4]\n"   \
  "stp q22, q23, [%[ptr_out0]], #32\n" \
  "fmla v31.8h, %[w6].8h, v1.h[6]\n"   \
  /* r4-1 */                           \
  "fmla v24.8h, %[w7].8h, v0.h[1]\n"   \
  "fmla v25.8h, %[w7].8h, v0.h[3]\n"   \
  "fmla v26.8h, %[w7].8h, v0.h[5]\n"   \
  "fmla v27.8h, %[w7].8h, v0.h[7]\n"   \
  "fmla v28.8h, %[w7].8h, v1.h[1]\n"   \
  "fmla v29.8h, %[w7].8h, v1.h[3]\n"   \
  "fmla v30.8h, %[w7].8h, v1.h[5]\n"   \
  "fmla v31.8h, %[w7].8h, v1.h[7]\n"   \
  /* r4-2 */                           \
  "fmla v24.8h, %[w8].8h, v0.h[2]\n"   \
  "fmla v25.8h, %[w8].8h, v0.h[4]\n"   \
  "fmla v26.8h, %[w8].8h, v0.h[6]\n"   \
  "subs   %w[cnt], %w[cnt], #1\n"      \
  "fmla v27.8h, %[w8].8h, v1.h[0]\n"   \
  "stp q24, q25, [%[ptr_out1]], #32\n" \
  "fmla v28.8h, %[w8].8h, v1.h[2]\n"   \
  "fmla v29.8h, %[w8].8h, v1.h[4]\n"   \
  "stp q26, q27, [%[ptr_out1]], #32\n" \
  "fmla v30.8h, %[w8].8h, v1.h[6]\n"   \
  "fmla v31.8h, %[w8].8h, v10.h[0]\n"  \
  "stp q28, q29, [%[ptr_out1]], #32\n" \
  "stp q30, q31, [%[ptr_out1]], #32\n" \
  "bne    2b\n"
#define ASM_PARAM                                  \
  : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), \
    [r2] "+r"(r2), [r3] "+r"(r3), [r4] "+r"(r4),   \
    [ptr_out0] "+r"(ptr_out0),                     \
    [ptr_out1] "+r"(ptr_out1)                      \
  : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2),      \
    [w3] "w"(w3), [w4] "w"(w4), [w5] "w"(w5),      \
    [w6] "w"(w6), [w7] "w"(w7), [w8] "w"(w8)       \
  : "cc", "memory", "v0", "v1", "v4", "v5", "v10", \
    "v12", "v16", "v17", "v18", "v19", "v20",      \
    "v21", "v22", "v23", "v24", "v25", "v26",      \
    "v27", "v28", "v29", "v30", "v31"

#define COMPUTE_LINE \
  /* 00 c0*/\
  "fmla v24.8h, v17.8h, v8.h[0]\n" \
  "fmla v25.8h, v17.8h, v9.h[0]\n" \
  "fmla v26.8h, v17.8h, v10.h[0]\n"\
  "fmla v27.8h, v17.8h, v11.h[0]\n"\
  "fmla v28.8h, v17.8h, v12.h[0]\n"\
  "fmla v29.8h, v17.8h, v13.h[0]\n"\
  "fmla v30.8h, v17.8h, v14.h[0]\n"\
  "fmla v31.8h, v17.8h, v15.h[0]\n"\
  /* c1*/\
  "fmla v24.8h, v18.8h, v8.h[1]\n" \
  "fmla v25.8h, v18.8h, v9.h[1]\n" \
  "fmla v26.8h, v18.8h, v10.h[1]\n"\
  "fmla v27.8h, v18.8h, v11.h[1]\n"\
  "fmla v28.8h, v18.8h, v12.h[1]\n"\
  "fmla v29.8h, v18.8h, v13.h[1]\n"\
  "fmla v30.8h, v18.8h, v14.h[1]\n"\
  "fmla v31.8h, v18.8h, v15.h[1]\n"\
  /* c2*/\
  "fmla v24.8h, v19.8h, v8.h[2]\n" \
  "fmla v25.8h, v19.8h, v9.h[2]\n" \
  "fmla v26.8h, v19.8h, v10.h[2]\n"\
  "fmla v27.8h, v19.8h, v11.h[2]\n"\
  "fmla v28.8h, v19.8h, v12.h[2]\n"\
  "fmla v29.8h, v19.8h, v13.h[2]\n"\
  "fmla v30.8h, v19.8h, v14.h[2]\n"\
  "fmla v31.8h, v19.8h, v15.h[2]\n"\
  /* 01 c0*/\
  "fmla v24.8h, v20.8h, v8.h[4]\n" \
  "fmla v25.8h, v20.8h, v9.h[4]\n" \
  "fmla v26.8h, v20.8h, v10.h[4]\n"\
  "fmla v27.8h, v20.8h, v11.h[4]\n"\
  "fmla v28.8h, v20.8h, v12.h[4]\n"\
  "fmla v29.8h, v20.8h, v13.h[4]\n"\
  "fmla v30.8h, v20.8h, v14.h[4]\n"\
  "fmla v31.8h, v20.8h, v15.h[4]\n"\
  /* c1*/\
  "fmla v24.8h, v21.8h, v8.h[5]\n" \
  "fmla v25.8h, v21.8h, v9.h[5]\n" \
  "fmla v26.8h, v21.8h, v10.h[5]\n"\
  "fmla v27.8h, v21.8h, v11.h[5]\n"\
  "fmla v28.8h, v21.8h, v12.h[5]\n"\
  "fmla v29.8h, v21.8h, v13.h[5]\n"\
  "fmla v30.8h, v21.8h, v14.h[5]\n"\
  "fmla v31.8h, v21.8h, v15.h[5]\n"

#define COMPUTE_C3                   \
  "ldr   q8,  [%[r0]]\n"             \
  "ldr   q17, [%[wc]]\n"             \
  "ldr   q9,  [%[r0], #0x10]\n"      \
  "ldr   q18,       [%[wc], #0x10]\n"\
  "ldr   q10,  [%[r0], #0x20]\n"      \
  "ldr   q19,       [%[wc], #0x20]\n"\
  "ldr   q11,  [%[r0], #0x30]\n"      \
  "ldr   q20,       [%[wc], #0x30]\n"\
  "ldr   q12,  [%[r0], #0x40]\n"      \
  "1:      \n"                       \
  /* line 0 0 c0*/                   \
  "fmul v24.8h, %[w0].8h, v8.h[0]\n" \
  "ldr   q13,  [%[r0], #0x50]\n"     \
  "fmul v25.8h, %[w0].8h, v9.h[0]\n" \
  "ldr   q14,  [%[r0], #0x60]\n"     \
  "fmul v26.8h, %[w0].8h, v10.h[0]\n"\
  "ldr   q15,  [%[r0], #0x70]\n"     \
  "fmul v27.8h, %[w0].8h, v11.h[0]\n"\
  "fmul v28.8h, %[w0].8h, v12.h[0]\n"\
  "ldr   q21,       [%[wc], #0x40]\n"\
  "fmul v29.8h, %[w0].8h, v13.h[0]\n"\
  "ldr   q22,       [%[wc], #0x50]\n"\
  "fmul v30.8h, %[w0].8h, v14.h[0]\n"\
  "fmul v31.8h, %[w0].8h, v15.h[0]\n"\
  "add  %[r0],  %[r0],    #0x80\n"   \
  /* c1*/\
  "fmla v24.8h, %[w1].8h, v8.h[1]\n" \
  "fmla v25.8h, %[w1].8h, v9.h[1]\n" \
  "fmla v26.8h, %[w1].8h, v10.h[1]\n"\
  "fmla v27.8h, %[w1].8h, v11.h[1]\n"\
  "fmla v28.8h, %[w1].8h, v12.h[1]\n"\
  "fmla v29.8h, %[w1].8h, v13.h[1]\n"\
  "fmla v30.8h, %[w1].8h, v14.h[1]\n"\
  "fmla v31.8h, %[w1].8h, v15.h[1]\n"\
  /* c2*/\
  "fmla v24.8h, %[w2].8h, v8.h[2]\n" \
  "fmla v25.8h, %[w2].8h, v9.h[2]\n" \
  "fmla v26.8h, %[w2].8h, v10.h[2]\n"\
  "fmla v27.8h, %[w2].8h, v11.h[2]\n"\
  "fmla v28.8h, %[w2].8h, v12.h[2]\n"\
  "fmla v29.8h, %[w2].8h, v13.h[2]\n"\
  "fmla v30.8h, %[w2].8h, v14.h[2]\n"\
  "fmla v31.8h, %[w2].8h, v15.h[2]\n"\
  /* line 0 01 c0*/                  \
  "fmla v24.8h, %[w3].8h, v8.h[4]\n" \
  "fmla v25.8h, %[w3].8h, v9.h[4]\n" \
  "fmla v26.8h, %[w3].8h, v10.h[4]\n"\
  "fmla v27.8h, %[w3].8h, v11.h[4]\n"\
  "fmla v28.8h, %[w3].8h, v12.h[4]\n"\
  "fmla v29.8h, %[w3].8h, v13.h[4]\n"\
  "fmla v30.8h, %[w3].8h, v14.h[4]\n"\
  "fmla v31.8h, %[w3].8h, v15.h[4]\n"\
  /* c1*/\
  "fmla v24.8h, %[w4].8h, v8.h[5]\n" \
  "fmla v25.8h, %[w4].8h, v9.h[5]\n" \
  "fmla v26.8h, %[w4].8h, v10.h[5]\n"\
  "fmla v27.8h, %[w4].8h, v11.h[5]\n"\
  "fmla v28.8h, %[w4].8h, v12.h[5]\n"\
  "fmla v29.8h, %[w4].8h, v13.h[5]\n"\
  "fmla v30.8h, %[w4].8h, v14.h[5]\n"\
  "fmla v31.8h, %[w4].8h, v15.h[5]\n"\
  /* c2*/\
  "fmla v24.8h, %[w5].8h, v8.h[6]\n" \
  "fmla v25.8h, %[w5].8h, v9.h[6]\n" \
  "fmla v26.8h, %[w5].8h, v10.h[6]\n"\
  "fmla v27.8h, %[w5].8h, v11.h[6]\n"\
  "fmla v28.8h, %[w5].8h, v12.h[6]\n"\
  "fmla v29.8h, %[w5].8h, v13.h[6]\n"\
  "fmla v30.8h, %[w5].8h, v14.h[6]\n"\
  "fmla v31.8h, %[w5].8h, v15.h[6]\n"\
  /* line 0 02 c0*/                  \
  "ldr   q7,  [%[r0], #0x00]\n"      \
  "fmla v24.8h, %[w6].8h, v9.h[0]\n" \
  "fmla v25.8h, %[w6].8h, v10.h[0]\n"\
  "ldr    q8,   [%[r1]]\n"           \
  "fmla v26.8h, %[w6].8h, v11.h[0]\n"\
  "fmla v27.8h, %[w6].8h, v12.h[0]\n"\
  "fmla v28.8h, %[w6].8h, v13.h[0]\n"\
  "fmla v29.8h, %[w6].8h, v14.h[0]\n"\
  "fmla v30.8h, %[w6].8h, v15.h[0]\n"\
  "fmla v31.8h, %[w6].8h, v7.h[0]\n"\
  /* c1*/\
  "fmla v24.8h, %[w7].8h, v9.h[1]\n" \
  "fmla v25.8h, %[w7].8h, v10.h[1]\n"\
  "fmla v26.8h, %[w7].8h, v11.h[1]\n"\
  "fmla v27.8h, %[w7].8h, v12.h[1]\n"\
  "fmla v28.8h, %[w7].8h, v13.h[1]\n"\
  "fmla v29.8h, %[w7].8h, v14.h[1]\n"\
  "fmla v30.8h, %[w7].8h, v15.h[1]\n"\
  "fmla v31.8h, %[w7].8h, v7.h[1]\n"\
  /* c2*/\
  "fmla v24.8h, %[w8].8h, v9.h[2]\n" \
  "ldr    q9,   [%[r1], #0x10]\n"    \
  "fmla v25.8h, %[w8].8h, v10.h[2]\n"\
  "ldr    q10,   [%[r1], #0x20]\n"   \
  "fmla v26.8h, %[w8].8h, v11.h[2]\n"\
  "ldr    q11,   [%[r1], #0x30]\n"   \
  "fmla v27.8h, %[w8].8h, v12.h[2]\n"\
  "ldr    q12,   [%[r1], #0x40]\n"   \
  "fmla v28.8h, %[w8].8h, v13.h[2]\n"\
  "ldr    q13,   [%[r1], #0x50]\n"   \
  "fmla v29.8h, %[w8].8h, v14.h[2]\n"\
  "ldr    q14,   [%[r1], #0x60]\n"   \
  "fmla v30.8h, %[w8].8h, v15.h[2]\n"\
  "ldr    q15,   [%[r1], #0x70]\n"   \
  "fmla v31.8h, %[w8].8h, v7.h[2]\n" \
  "add   %[r1],  %[r1],  #0x80\n"    \
  /* line 1 */ \
  COMPUTE_LINE \
  /* c2*/\
  "ldr   q17,   [%[wc], #0x60]\n"  \
  "fmla v24.8h, v22.8h, v8.h[6]\n" \
  "fmla v25.8h, v22.8h, v9.h[6]\n" \
  "fmla v26.8h, v22.8h, v10.h[6]\n"\
  "fmla v27.8h, v22.8h, v11.h[6]\n"\
  "ldr    q7,   [%[r1]]\n"         \
  "fmla v28.8h, v22.8h, v12.h[6]\n"\
  "fmla v29.8h, v22.8h, v13.h[6]\n"\
  "fmla v30.8h, v22.8h, v14.h[6]\n"\
  "fmla v31.8h, v22.8h, v15.h[6]\n"\
  /* 02 c0*/\
  "ldr   q18,   [%[wc], #0x70]\n"  \
  "fmla v24.8h, v17.8h, v9.h[0]\n" \
  "fmla v25.8h, v17.8h, v10.h[0]\n"\
  "fmla v26.8h, v17.8h, v11.h[0]\n"\
  "fmla v27.8h, v17.8h, v12.h[0]\n"\
  "fmla v28.8h, v17.8h, v13.h[0]\n"\
  "fmla v29.8h, v17.8h, v14.h[0]\n"\
  "fmla v30.8h, v17.8h, v15.h[0]\n"\
  "fmla v31.8h, v17.8h, v7.h[0]\n"\
  /* c1*/\
  "ldr   q19,   [%[wc], #0x80]\n"  \
  "fmla v24.8h, v18.8h, v9.h[1]\n" \
  "ldr   q17,   [%[wc], #0x90]\n"  \
  "fmla v25.8h, v18.8h, v10.h[1]\n"\
  "ldr   q20,   [%[wc], #0xc0]\n"  \
  "fmla v26.8h, v18.8h, v11.h[1]\n"\
  "ldr   q21,   [%[wc], #0xd0]\n"  \
  "fmla v27.8h, v18.8h, v12.h[1]\n"\
  "ldr   q22,   [%[wc], #0xe0]\n"  \
  "fmla v28.8h, v18.8h, v13.h[1]\n"\
  "fmla v29.8h, v18.8h, v14.h[1]\n"\
  "fmla v30.8h, v18.8h, v15.h[1]\n"\
  "fmla v31.8h, v18.8h, v7.h[1]\n"\
  /* c2*/\
  "ldr   q18,   [%[wc], #0xa0]\n"  \
  "ldr    q8,   [%[r2], #0x00]\n"  \
  "fmla v24.8h, v19.8h, v9.h[2]\n" \
  "ldr    q9,   [%[r2], #0x10]\n"  \
  "fmla v25.8h, v19.8h, v10.h[2]\n"\
  "ldr    q10,  [%[r2], #0x20]\n"  \
  "fmla v26.8h, v19.8h, v11.h[2]\n"\
  "ldr    q11,  [%[r2], #0x30]\n"  \
  "fmla v27.8h, v19.8h, v12.h[2]\n"\
  "ldr    q12,  [%[r2], #0x40]\n"  \
  "fmla v28.8h, v19.8h, v13.h[2]\n"\
  "ldr    q13,  [%[r2], #0x50]\n"  \
  "fmla v29.8h, v19.8h, v14.h[2]\n"\
  "ldr    q14,  [%[r2], #0x60]\n"  \
  "fmla v30.8h, v19.8h, v15.h[2]\n"\
  "ldr    q15,  [%[r2], #0x70]\n"  \
  "fmla v31.8h, v19.8h, v7.h[2]\n" \
  "add   %[r2],  %[r2],  #0x80\n"  \
  /* line 2 */ \
  "ldr   q19,   [%[wc], #0xb0]\n"  \
  COMPUTE_LINE \
  /* c2*/\
  "ldr   q17,   [%[wc], #0xf0]\n"  \
  "fmla v24.8h, v22.8h, v8.h[6]\n" \
  "fmla v25.8h, v22.8h, v9.h[6]\n" \
  "fmla v26.8h, v22.8h, v10.h[6]\n"\
  "fmla v27.8h, v22.8h, v11.h[6]\n"\
  "ldr    q7,   [%[r2]]\n"         \
  "fmla v28.8h, v22.8h, v12.h[6]\n"\
  "fmla v29.8h, v22.8h, v13.h[6]\n"\
  "fmla v30.8h, v22.8h, v14.h[6]\n"\
  "fmla v31.8h, v22.8h, v15.h[6]\n"\
  /* 02 c0*/\
  "ldr   q18,   [%[wc], #0x100]\n"  \
  "fmla v24.8h, v17.8h, v9.h[0]\n" \
  "fmla v25.8h, v17.8h, v10.h[0]\n"\
  "fmla v26.8h, v17.8h, v11.h[0]\n"\
  "fmla v27.8h, v17.8h, v12.h[0]\n"\
  "fmla v28.8h, v17.8h, v13.h[0]\n"\
  "fmla v29.8h, v17.8h, v14.h[0]\n"\
  "fmla v30.8h, v17.8h, v15.h[0]\n"\
  "fmla v31.8h, v17.8h, v7.h[0]\n"\
  /* c1*/\
  "ldr   q19,   [%[wc], #0x110]\n" \
  "fmla v24.8h, v18.8h, v9.h[1]\n" \
  "fmla v25.8h, v18.8h, v10.h[1]\n"\
  "fmla v26.8h, v18.8h, v11.h[1]\n"\
  "ldr   q20,   [%[wc], #0x30]\n"  \
  "fmla v27.8h, v18.8h, v12.h[1]\n"\
  "fmla v28.8h, v18.8h, v13.h[1]\n"\
  "fmla v29.8h, v18.8h, v14.h[1]\n"\
  "fmla v30.8h, v18.8h, v15.h[1]\n"\
  "fmla v31.8h, v18.8h, v7.h[1]\n"\
  "ldr    q8,   [%[r0], #0x00]\n"  \
  "fmla v24.8h, v19.8h, v9.h[2]\n" \
  "ldr    q9,   [%[r0], #0x10]\n"  \
  "fmla v25.8h, v19.8h, v10.h[2]\n"\
  "ldr    q10,  [%[r0], #0x20]\n"  \
  "fmla v26.8h, v19.8h, v11.h[2]\n"\
  "ldr    q11,  [%[r0], #0x30]\n"  \
  "fmla v27.8h, v19.8h, v12.h[2]\n"\
  "ldr    q12,  [%[r0], #0x40]\n"  \
  "subs  %w[cnt], %w[cnt], #1\n"   \
  "str  q24, [%[ptr_out0], #0x00]\n"\
  "fmla v28.8h, v19.8h, v13.h[2]\n"\
  "str  q25, [%[ptr_out0], #0x10]\n"\
  "fmla v29.8h, v19.8h, v14.h[2]\n"\
  "str  q26, [%[ptr_out0], #0x20]\n"\
  "fmla v30.8h, v19.8h, v15.h[2]\n"\
  "str  q27, [%[ptr_out0], #0x30]\n"\
  "fmla v31.8h, v19.8h, v7.h[2]\n"\
  "str  q28, [%[ptr_out0], #0x40]\n"\
  "ldr   q17,   [%[wc], #0x00]\n"  \
  "str  q29, [%[ptr_out0], #0x50]\n"\
  "ldr   q18,   [%[wc], #0x10]\n"  \
  "str  q30, [%[ptr_out0], #0x60]\n"\
  "ldr   q19,   [%[wc], #0x20]\n"  \
  "str  q31, [%[ptr_out0], #0x70]\n"\
  "add  %[ptr_out0], %[ptr_out0], #0x80\n"\
  "bne  1b\n"
#else

#define INIT_FIRST                                                             \
  "2:\n"                                                                       \
  "vld1.16    {d10-d13}, [%[wc0]]!       @ load w0, w1\n"                      \
  "vld1.16    {d0-d2}, [%[r0]]           @ load r0\n"                          \
  "add    %[r0], %[r0], #16\n"                                                        \
  "vmul.f16   q8, q5, d0[0]              @ w0 * inr00\n"                       \
  "vmul.f16   q9, q5, d0[2]              @ w0 * inr02\n"                       \
  "vmul.f16   q10, q5, d1[0]             @ w0 * inr04\n"                       \
  "vmul.f16   q11, q5, d1[2]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  "vld1.16    {d3-d5}, [%[r2]]           @ load r2\n"                          \
  "add    %[r2], %[r2], #16\n"                                                        \
  "vmul.f16   q12, q5, d3[0]             @ w0 * inr20\n"                       \
  "vmul.f16   q13, q5, d3[2]             @ w0 * inr22\n"                       \
  "vld1.16    {d14-d15}, [%[wc0]]!       @ load w2\n"                          \  
  "vmul.f16   q14, q5, d4[0]             @ w0 * inr24\n"                       \
  "vmul.f16   q15, q5, d4[2]             @ w0 * inr26\n"

#define INIT                          \
  "2:\n"                              \
  "vld1.16    {d10-d13}, [%[wc0]]!       @ load w0, w1\n"                      \
  "vld1.16    {d16-d19}, [%[ptr_out0]]!   @ load outr0\n"                      \
  "vld1.16    {d0-d2}, [%[r0]]          @ load r0\n"                           \
  "add    %[r0], %[r0], #16\n"                                                        \
  "vld1.16    {d14-d15}, [%[wc0]]!       @ load w2\n"                          \  
  "vmla.f16   q8, q5, d0[0]              @ w0 * inr00\n"                       \
  "vld1.16    {d20-d23}, [%[ptr_out0]]    @ load outr0\n"                      \  
  "sub    %[ptr_out0], %[ptr_out0], #32\n"                                     \  
  "vmla.f16   q9, q5, d0[2]              @ w0 * inr02\n"                       \
  "vmla.f16   q10, q5, d1[0]             @ w0 * inr04\n"                       \
  "vld1.16    {d24-d27}, [%[ptr_out1]]!   @ load outr0\n"                      \  
  "vmla.f16   q11, q5, d1[2]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  "vld1.16    {d3-d5}, [%[r2]]          @ load r2\n"                           \
  "add    %[r2], %[r2], #16\n"                                                        \
  "vmla.f16   q12, q5, d3[0]             @ w0 * inr20\n"                       \
  "vmla.f16   q13, q5, d3[2]             @ w0 * inr22\n"                       \
  "vld1.16    {d28-d31}, [%[ptr_out1]]    @ load outr0\n"                      \
  "sub    %[ptr_out1], %[ptr_out1], #32\n"                                     \  
  "vmla.f16   q14, q5, d4[0]             @ w0 * inr24\n"                       \
  "vmla.f16   q15, q5, d4[2]             @ w0 * inr26\n"

#define COMPUTE                        \
  /* r0-1 */                           \
  "vld1.16    {d6-d8}, [%[r1]]           @ load r1\n"                          \
  "add  %[r1], %[r1], #16\n"                                                   \
  "vmla.f16   q8, q6, d0[1]              @ w0 * inr00\n"                       \
  "vmla.f16   q9, q6, d0[3]              @ w0 * inr02\n"                       \
  "vmla.f16   q10, q6, d1[1]             @ w0 * inr04\n"                       \
  "vmla.f16   q11, q6, d1[3]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  /* r2-1 */                           \
  "vmla.f16   q12, q6, d3[1]             @ w0 * inr20\n"                       \
  "vmla.f16   q13, q6, d3[3]             @ w0 * inr22\n"                       \
  "vmla.f16   q14, q6, d4[1]             @ w0 * inr24\n"                       \
  "vmla.f16   q15, q6, d4[3]             @ w0 * inr26\n"                       \
  "vld1.16    {d10-d13}, [%[wc0]]!       @ load w5, to q7\n" /* mul r1, with*/ \  
  /* r0-2 */                           \
  "vmla.f16   q8, q7, d0[2]              @ w0 * inr00\n"                       \
  "vmla.f16   q9, q7, d1[0]              @ w0 * inr02\n"                       \
  "vmla.f16   q10, q7, d1[2]             @ w0 * inr04\n"                       \
  "vmla.f16   q11, q7, d2[0]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  /* r2-2 */                           \
  "vmla.f16   q12, q7, d3[2]             @ w0 * inr20\n"                       \
  "vmla.f16   q13, q7, d4[0]             @ w0 * inr22\n"                       \
  "vmla.f16   q14, q7, d4[2]             @ w0 * inr24\n"                       \
  "vmla.f16   q15, q7, d5[0]             @ w0 * inr26\n"                       \
  "vld1.16    {d14-d15}, [%[wc0]]!       @ load w5, to q7\n" /* mul r1, with*/ \
  /* r1-0 */                                                                   \
  "vmla.f16   q8, q5, d6[0]              @ w0 * inr00\n"                       \
  "vmla.f16   q9, q5, d6[2]              @ w0 * inr02\n"                       \
  "vmla.f16   q10, q5, d7[0]             @ w0 * inr04\n"                       \
  "vmla.f16   q11, q5, d7[2]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  "vld1.16    {d0-d2}, [%[r3]]           @ load r1\n"                          \
  "add  %[r3], %[r3], #16\n"                                                   \  
  /* r1-1 */                                                                   \
  "vmla.f16   q8, q6, d6[1]              @ w0 * inr00\n"                       \
  "vmla.f16   q9, q6, d6[3]              @ w0 * inr02\n"                       \
  "vmla.f16   q10, q6, d7[1]             @ w0 * inr04\n"                       \
  "vmla.f16   q11, q6, d7[3]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  /* r1-2 */                                                                   \
  "vmla.f16   q8, q7, d6[2]              @ w0 * inr00\n"                       \
  "vmla.f16   q9, q7, d7[0]              @ w0 * inr02\n"                       \
  "vmla.f16   q10, q7, d7[2]             @ w0 * inr04\n"                       \
  "vmov d6, d8 \n"                \
  "vmla.f16   q11, q7, d6[0]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  /* r3-0 */                                                                   \
  "vmla.f16   q12, q5, d0[0]              @ w0 * inr00\n"                       \
  "vmla.f16   q13, q5, d0[2]              @ w0 * inr02\n"                       \
  "vmla.f16   q14, q5, d1[0]             @ w0 * inr04\n"                       \
  "vmla.f16   q15, q5, d1[2]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  /* r3-1 */                                                                   \
  "vmla.f16   q12, q6, d0[1]              @ w0 * inr00\n"                       \
  "vmla.f16   q13, q6, d0[3]              @ w0 * inr02\n"                       \
  "vmla.f16   q14, q6, d1[1]             @ w0 * inr04\n"                       \
  "vmla.f16   q15, q6, d1[3]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  "vld1.16    {d10-d13}, [%[wc0]]!       @ load w0, w1\n"                      \
  /* r3-2 */                                                                   \
  "vmla.f16   q12, q7, d0[2]              @ w0 * inr00\n"                       \
  "vmla.f16   q13, q7, d1[0]              @ w0 * inr02\n"                       \
  "vmla.f16   q14, q7, d1[2]             @ w0 * inr04\n"                       \
  "vmla.f16   q15, q7, d2[0]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  "vld1.16    {d6-d8}, [%[r4]]           @ load r3\n"                          \
  "add  %[r4], %[r4], #16\n"                                                          \
  "vld1.32    {d14-d15}, [%[wc0]]!       @ load w5, to q7\n" /* mul r1, with*/ \
  "sub    %[wc0], %[wc0], #144\n"                                                \
  /* r2-0 */                                                                   \
  "vmla.f16   q8, q5, d3[0]              @ w0 * inr00\n"                       \
  "vmla.f16   q9, q5, d3[2]              @ w0 * inr02\n"                       \
  "vmla.f16   q10, q5, d4[0]             @ w0 * inr04\n"                       \
  "vmla.f16   q11, q5, d4[2]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  /* r2-1 */                                                                   \
  "vmla.f16   q8, q6, d3[1]              @ w0 * inr00\n"                       \
  "vmla.f16   q9, q6, d3[3]              @ w0 * inr02\n"                       \
  "vmla.f16   q10, q6, d4[1]             @ w0 * inr04\n"                       \
  "vmla.f16   q11, q6, d4[3]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  /* r2-2 */                                                                   \
  "vmla.f16   q8, q7, d3[2]              @ w0 * inr00\n"                       \
  "vmla.f16   q9, q7, d4[0]              @ w0 * inr02\n"                       \
  "vmla.f16   q10, q7, d4[2]             @ w0 * inr04\n"                       \
  "vmla.f16   q11, q7, d5[0]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  /* r4-0 */                                                                   \
  "vmla.f16   q12, q5, d6[0]              @ w0 * inr00\n"                       \
  "vmla.f16   q13, q5, d6[2]              @ w0 * inr02\n"                       \
  "vmla.f16   q14, q5, d7[0]             @ w0 * inr04\n"                       \
  "vmla.f16   q15, q5, d7[2]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  /* r4-1 */                                                                   \
  "vmla.f16   q12, q6, d6[1]              @ w0 * inr00\n"                       \
  "vmla.f16   q13, q6, d6[3]              @ w0 * inr02\n"                       \
  "vmla.f16   q14, q6, d7[1]             @ w0 * inr04\n"                       \
  "vmla.f16   q15, q6, d7[3]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  /* r4-2 */                                                                   \
  "vmla.f16   q12, q7, d6[2]              @ w0 * inr00\n"                       \
  "vmla.f16   q13, q7, d7[0]              @ w0 * inr02\n"                       \
  "vmla.f16   q14, q7, d7[2]             @ w0 * inr04\n"                       \
  "vmov d6, d8 \n"                \
  "vmla.f16   q15, q7, d6[0]             @ w0 * inr06\n" /* mul r0, with w0*/  \
  "vst1.16    {d16-d19}, [%[ptr_out0]]!  @ load outr0\n"                       \
  "vst1.16    {d20-d23}, [%[ptr_out0]]!   @ load outr0\n"                       \
  "vst1.16    {d24-d27}, [%[ptr_out1]]!  @ save r10, r11\n"                    \
  "vst1.16    {d28-d31}, [%[ptr_out1]]!  @ save r12, r13\n"                    \
  "subs  %[cnt], %[cnt], #1\n"                                               \
  "bne    2b\n"
#define ASM_PARAM                                  \
  : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), \
    [r2] "+r"(r2), [r3] "+r"(r3), [r4] "+r"(r4),   \
    [wc0] "+r"(wc0),                               \
    [ptr_out0] "+r"(ptr_out0),                     \
    [ptr_out1] "+r"(ptr_out1)                      \
  :                                                \
  : "cc", "memory", "q0", "q1", "q2", "q3", "q4",  \
    "q5", "q6", "q7", "q8", "q9", "q10",           \
    "q11", "q12", "q13", "q14", "q15"

#define COMPUTE_C3                                          \
  "1: \n"                                                   \
  "vld1.16  {d0-d3}, [%[r0]]!          @ load q0, q1\n"     \
  "vld1.16  {d18-d19}, [%[wc0]]!       @ load w0, w1\n"     \
  "vld1.16  {d4-d7}, [%[r0]]!          @ load q2, q3\n"     \
  "vld1.16  {d20-d21}, [%[wc0]]!       @ load w2, w3\n"     \
  "vld1.16  {d8}, [%[r0]]          @ load q4, q5\n"         \
  "vld1.16  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"     \
  /* line 0*/                                               \
  /* line 0 0 c0*/                                          \
  "vmul.f16   q15, q9, d6[0]          @ mul \n"            \
  "vmul.f16   q12, q9, d0[0]           @ mul \n"            \
  "vmul.f16   q13, q9, d2[0]           @ mul \n"            \
  "vmul.f16   q14, q9, d4[0]           @ mul \n"            \
  "vld1.16  {d18-d19}, [%[wc0]]!       @ load w0, w1\n"     \
  /* c1*/\
  "vmla.f16   q15, q10, d6[1]          @ mul \n"            \
  "vmla.f16   q12, q10, d0[1]           @ mul \n"            \
  "vmla.f16   q13, q10, d2[1]           @ mul \n"            \
  "vmla.f16   q14, q10, d4[1]           @ mul \n"            \
  "vld1.16  {d20-d21}, [%[wc0]]!       @ load w2, w3\n"     \
  /* c2*/\
  "vmla.f16   q15, q11, d6[2]          @ mul \n"            \
  "vmla.f16   q12, q11, d0[2]           @ mul \n"            \
  "vmla.f16   q13, q11, d2[2]           @ mul \n"            \
  "vmla.f16   q14, q11, d4[2]           @ mul \n"            \
  "vld1.16  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"     \
  /* line 0 01 c0*/                  \
  "vmla.f16   q15, q9, d7[0]          @ mul \n"            \
  "vmla.f16   q12, q9, d1[0]           @ mul \n"            \
  "vmla.f16   q13, q9, d3[0]           @ mul \n"            \
  "vmla.f16   q14, q9, d5[0]           @ mul \n"            \
  "vld1.16  {d18-d19}, [%[wc0]]!       @ load w0, w1\n"     \
  /* c1*/\
  "vmla.f16   q15, q10, d7[1]          @ mul \n"            \
  "vmla.f16   q12, q10, d1[1]           @ mul \n"            \
  "vmla.f16   q13, q10, d3[1]           @ mul \n"            \
  "vmla.f16   q14, q10, d5[1]           @ mul \n"            \
  "vld1.16  {d20-d21}, [%[wc0]]!       @ load w2, w3\n"     \
  /* c2*/\
  "vmla.f16   q15, q11, d7[2]          @ mul \n"            \
  "vmla.f16   q12, q11, d1[2]           @ mul \n"            \
  "vmla.f16   q13, q11, d3[2]           @ mul \n"            \
  "vmla.f16   q14, q11, d5[2]           @ mul \n"            \
  "vld1.16  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"     \
  /* line 0 02 c0*/                  \
  "vmov d0, d8 \n"                \
  "vmla.f16   q15, q9, d0[0]          @ mul \n"            \
  "vmla.f16   q12, q9, d2[0]           @ mul \n"            \
  "vmla.f16   q13, q9, d4[0]           @ mul \n"            \
  "vmla.f16   q14, q9, d6[0]           @ mul \n"            \
  "vld1.16  {d18-d19}, [%[wc0]]!       @ load w0, w1\n"     \
  /* c1*/\
  "vmla.f16   q15, q10, d0[1]          @ mul \n"            \
  "vmla.f16   q12, q10, d2[1]           @ mul \n"            \
  "vmla.f16   q13, q10, d4[1]           @ mul \n"            \
  "vmla.f16   q14, q10, d6[1]           @ mul \n"            \
  "vld1.16  {d20-d21}, [%[wc0]]!       @ load w2, w3\n"     \
  /* c2*/\
  "vmla.f16   q15, q11, d0[2]          @ mul \n"            \
  "vmla.f16   q12, q11, d2[2]           @ mul \n"            \
  "vmla.f16   q13, q11, d4[2]           @ mul \n"            \
  "vmla.f16   q14, q11, d6[2]           @ mul \n"            \
  "vld1.16  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"     \
  /*i1*/\
  "vld1.16  {d0-d3}, [%[r1]]!          @ load q0, q1\n"     \
  "vld1.16  {d4-d7}, [%[r1]]!          @ load q2, q3\n"     \
  "vld1.16  {d8}, [%[r1]]          @ load q4, q5\n"     \
   /* line 0*/                                               \
  /* line 0 0 c0*/                                          \
  "vmla.f16   q15, q9, d6[0]          @ mul \n"            \
  "vmla.f16   q12, q9, d0[0]           @ mul \n"            \
  "vmla.f16   q13, q9, d2[0]           @ mul \n"            \
  "vmla.f16   q14, q9, d4[0]           @ mul \n"            \
  "vld1.16  {d18-d19}, [%[wc0]]!       @ load w0, w1\n"     \
  /* c1*/\
  "vmla.f16   q15, q10, d6[1]          @ mul \n"            \
  "vmla.f16   q12, q10, d0[1]           @ mul \n"            \
  "vmla.f16   q13, q10, d2[1]           @ mul \n"            \
  "vmla.f16   q14, q10, d4[1]           @ mul \n"            \
  "vld1.16  {d20-d21}, [%[wc0]]!       @ load w2, w3\n"     \
  /* c2*/\
  "vmla.f16   q15, q11, d6[2]          @ mul \n"            \
  "vmla.f16   q12, q11, d0[2]           @ mul \n"            \
  "vmla.f16   q13, q11, d2[2]           @ mul \n"            \
  "vmla.f16   q14, q11, d4[2]           @ mul \n"            \
  "vld1.16  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"     \
  /* line 0 01 c0*/                  \
  "vmla.f16   q15, q9, d7[0]          @ mul \n"            \
  "vmla.f16   q12, q9, d1[0]           @ mul \n"            \
  "vmla.f16   q13, q9, d3[0]           @ mul \n"            \
  "vmla.f16   q14, q9, d5[0]           @ mul \n"            \
  "vld1.16  {d18-d19}, [%[wc0]]!       @ load w0, w1\n"     \
  /* c1*/\
  "vmla.f16   q15, q10, d7[1]          @ mul \n"            \
  "vmla.f16   q12, q10, d1[1]           @ mul \n"            \
  "vmla.f16   q13, q10, d3[1]           @ mul \n"            \
  "vmla.f16   q14, q10, d5[1]           @ mul \n"            \
  "vld1.16  {d20-d21}, [%[wc0]]!       @ load w2, w3\n"     \
  /* c2*/\
  "vmla.f16   q15, q11, d7[2]          @ mul \n"            \
  "vmla.f16   q12, q11, d1[2]           @ mul \n"            \
  "vmla.f16   q13, q11, d3[2]           @ mul \n"            \
  "vmla.f16   q14, q11, d5[2]           @ mul \n"            \
  "vld1.16  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"     \
  /* line 0 02 c0*/                  \
  "vmov d0, d8 \n"                \
  "vmla.f16   q15, q9, d0[0]          @ mul \n"            \
  "vmla.f16   q12, q9, d2[0]           @ mul \n"            \
  "vmla.f16   q13, q9, d4[0]           @ mul \n"            \
  "vmla.f16   q14, q9, d6[0]           @ mul \n"            \
  "vld1.16  {d18-d19}, [%[wc0]]!       @ load w0, w1\n"     \
  /* c1*/\
  "vmla.f16   q15, q10, d0[1]          @ mul \n"            \
  "vmla.f16   q12, q10, d2[1]           @ mul \n"            \
  "vmla.f16   q13, q10, d4[1]           @ mul \n"            \
  "vmla.f16   q14, q10, d6[1]           @ mul \n"            \
  "vld1.16  {d20-d21}, [%[wc0]]!       @ load w2, w3\n"     \
  /* c2*/\
  "vmla.f16   q15, q11, d0[2]          @ mul \n"            \
  "vmla.f16   q12, q11, d2[2]           @ mul \n"            \
  "vmla.f16   q13, q11, d4[2]           @ mul \n"            \
  "vmla.f16   q14, q11, d6[2]           @ mul \n"            \
  "vld1.16  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"     \
  /*i2*/\
  "vld1.16  {d0-d3}, [%[r2]]!          @ load q0, q1\n"     \
  "vld1.16  {d4-d7}, [%[r2]]!          @ load q2, q3\n"     \
  "vld1.16  {d8}, [%[r2]]          @ load q4, q5\n"         \
   /* line 0*/                                               \
  /* line 0 0 c0*/                                          \
  "vmla.f16   q15, q9, d6[0]          @ mul \n"            \
  "vmla.f16   q12, q9, d0[0]           @ mul \n"            \
  "vmla.f16   q13, q9, d2[0]           @ mul \n"            \
  "vmla.f16   q14, q9, d4[0]           @ mul \n"            \
  "vld1.16  {d18-d19}, [%[wc0]]!       @ load w0, w1\n"     \
  /* c1*/\
  "vmla.f16   q15, q10, d6[1]          @ mul \n"            \
  "vmla.f16   q12, q10, d0[1]           @ mul \n"            \
  "vmla.f16   q13, q10, d2[1]           @ mul \n"            \
  "vmla.f16   q14, q10, d4[1]           @ mul \n"            \
  "vld1.16  {d20-d21}, [%[wc0]]!       @ load w2, w3\n"     \
  /* c2*/\
  "vmla.f16   q15, q11, d6[2]          @ mul \n"            \
  "vmla.f16   q12, q11, d0[2]           @ mul \n"            \
  "vmla.f16   q13, q11, d2[2]           @ mul \n"            \
  "vmla.f16   q14, q11, d4[2]           @ mul \n"            \
  "vld1.16  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"     \
  /* line 0 01 c0*/                  \
  "vmla.f16   q15, q9, d7[0]          @ mul \n"            \
  "vmla.f16   q12, q9, d1[0]           @ mul \n"            \
  "vmla.f16   q13, q9, d3[0]           @ mul \n"            \
  "vmla.f16   q14, q9, d5[0]           @ mul \n"            \
  "vld1.16  {d18-d19}, [%[wc0]]!       @ load w0, w1\n"     \
  /* c1*/\
  "vmla.f16   q15, q10, d7[1]          @ mul \n"            \
  "vmla.f16   q12, q10, d1[1]           @ mul \n"            \
  "vmla.f16   q13, q10, d3[1]           @ mul \n"            \
  "vmla.f16   q14, q10, d5[1]           @ mul \n"            \
  "vld1.16  {d20-d21}, [%[wc0]]!       @ load w2, w3\n"     \
  /* c2*/\
  "vmla.f16   q15, q11, d7[2]          @ mul \n"            \
  "vmla.f16   q12, q11, d1[2]           @ mul \n"            \
  "vmla.f16   q13, q11, d3[2]           @ mul \n"            \
  "vmla.f16   q14, q11, d5[2]           @ mul \n"            \
  "vld1.16  {d22-d23}, [%[wc0]]!       @ load w2, w3\n"     \
  /* line 0 02 c0*/                  \
  "vmov d0, d8 \n"                \
  "vmla.f16   q15, q9, d0[0]          @ mul \n"            \
  "vmla.f16   q12, q9, d2[0]           @ mul \n"            \
  "vmla.f16   q13, q9, d4[0]           @ mul \n"            \
  "vmla.f16   q14, q9, d6[0]           @ mul \n"            \
  /* c1*/\
  "vmla.f16   q15, q10, d0[1]          @ mul \n"            \
  "vmla.f16   q12, q10, d2[1]           @ mul \n"            \
  "vmla.f16   q13, q10, d4[1]           @ mul \n"            \
  "vmla.f16   q14, q10, d6[1]           @ mul \n"            \
  /* c2*/\
  "vmla.f16   q15, q11, d0[2]          @ mul \n"             \
  "vmla.f16   q12, q11, d2[2]           @ mul \n"            \
  "vmla.f16   q13, q11, d4[2]           @ mul \n"            \
  "vmla.f16   q14, q11, d6[2]           @ mul \n"            \
  "sub %[wc0], %[wc0], #432\n"                               \
  "vst1.16    {d24-d27}, [%[ptr_out0]]!         \n"          \
  "vst1.16    {d28-d31}, [%[ptr_out0]]!         \n"          \
  "subs %[cnt], #1\n"                                       \
  "bne    1b\n"
#endif
// clang-format on
void conv_3x3s2_direct_fp16_c3(const float16_t* i_data,
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
                               ARMContext* ctx);
void conv_3x3s2_direct_fp16(const float16_t* i_data,
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
  if (ic == 3 && (oc % 8 == 0)) {
    conv_3x3s2_direct_fp16_c3(
        i_data, o_data, bs, oc, oh, ow, ic, ih, win, weights, bias, param, ctx);
    return;
  }
  auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  const int pad_w = paddings[2];
  const int pad_h = paddings[0];
  DIRECT_WORKSPACE_COMPUTE(ctx, 3, 2, ow, oh, ic, OUT_C_BLOCK, OUT_H_BLOCK)

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
#ifdef __aarch64__
  int w_loop = wout_round >> 3;
#else
  int w_loop = wout_round >> 2;
#endif

  int c_remain = oc - (oc / OUT_C_BLOCK) * OUT_C_BLOCK;
  int c_round_down = (oc / OUT_C_BLOCK) * OUT_C_BLOCK;
  if (c_remain) {
    c_round_down++;
  }

  int out_row_stride = OUT_C_BLOCK * wout_round;
  auto act_type = act_param.active_type;
  bool flag_bias = param.bias != nullptr;
  float16_t alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3 hardswish:4
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
      if (h + hout_r_block > oh) {
        h_kernel = oh - h;
      }
      int hs = h * 2 /*stride_h*/ - pad_h;
      int he = hs + h_kernel * 2 /*stride_h*/ + 1;
      prepack_input_nxw(
          din_batch, pre_din, 0, ic, hs, he, ws, we, ic, win, ih, ptr_zero);

      const float16_t* cblock_inr0 = pre_din;
      const float16_t* cblock_inr1 = cblock_inr0 + in_len;
      const float16_t* cblock_inr2 = cblock_inr1 + in_len;
      const float16_t* cblock_inr3 = cblock_inr2 + in_len;
      const float16_t* cblock_inr4 = cblock_inr3 + in_len;

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
        const float16_t* block_inr4 = cblock_inr4;

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
          const float16_t* inr3 = block_inr3;
          const float16_t* inr4 = block_inr4;

          float16_t* pre_out0 = pre_out + hk * out_row_stride;
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
            inr4 += win_round;
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
            inr4 += win_round;
          }
          block_inr0 = block_inr4;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
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

void conv_3x3s2_direct_fp16_c3(const float16_t* i_data,
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
  int in_channel = 4;
  DIRECT_WORKSPACE_COMPUTE(
      ctx, 3, 2, ow, oh, in_channel, OUT_C_BLOCK, OUT_H_BLOCK)

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
#ifdef __aarch64__
  int w_loop = wout_round >> 3;
#else
  int w_loop = wout_round >> 2;
#endif

  int c_remain = oc - (oc / OUT_C_BLOCK) * OUT_C_BLOCK;
  int c_round_down = (oc / OUT_C_BLOCK) * OUT_C_BLOCK;
  if (c_remain) {
    c_round_down++;
  }

  int out_row_stride = OUT_C_BLOCK * wout_round;
  auto act_type = act_param.active_type;
  bool flag_bias = param.bias != nullptr;
  float16_t alpha = 0.f;
  int flag_act = 0x00;
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
      if (h + hout_r_block > oh) {
        h_kernel = oh - h;
      }
      int hs = h * 2 /*stride_h*/ - pad_h;
      int he = hs + h_kernel * 2 /*stride_h*/ + 1;
      prepack_input_nxwc4(
          din_batch, pre_din, 0, ic, hs, he, ws, we, ic, win, ih, ptr_zero);

      const float16_t* cblock_inr0 = pre_din;
      const float16_t* cblock_inr1 = cblock_inr0 + in_len;
      const float16_t* cblock_inr2 = cblock_inr1 + in_len;

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

        const float16_t* weight_c = weights + c * w_stride;
        const float16_t* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = bias + c;
        }

        for (int hk = 0; hk < h_kernel; hk++) {
          const float16_t* wc0 = weight_c;

          const float16_t* inr0 = block_inr0;
          const float16_t* inr1 = block_inr1;
          const float16_t* inr2 = block_inr2;

          float16_t* pre_out0 = pre_out + hk * out_row_stride;
#ifdef __aarch64__
          int cnt = w_loop;
          float16x8_t w0 = vld1q_f16(wc0);
          float16x8_t w1 = vld1q_f16(wc0 + 8);
          float16x8_t w2 = vld1q_f16(wc0 + 16);
          float16x8_t w3 = vld1q_f16(wc0 + 24);
          float16x8_t w4 = vld1q_f16(wc0 + 32);
          float16x8_t w5 = vld1q_f16(wc0 + 40);
          float16x8_t w6 = vld1q_f16(wc0 + 48);
          float16x8_t w7 = vld1q_f16(wc0 + 56);
          float16x8_t w8 = vld1q_f16(wc0 + 64);
          const float16_t* wc00 = wc0 + 72;
          asm volatile(COMPUTE_C3
                       : [cnt] "+r"(cnt),
                         [r0] "+r"(inr0),
                         [r1] "+r"(inr1),
                         [r2] "+r"(inr2),
                         [wc] "+r"(wc00),
                         [ptr_out0] "+r"(pre_out0)
                       : [w0] "w"(w0),
                         [w1] "w"(w1),
                         [w2] "w"(w2),
                         [w3] "w"(w3),
                         [w4] "w"(w4),
                         [w5] "w"(w5),
                         [w6] "w"(w6),
                         [w7] "w"(w7),
                         [w8] "w"(w8)
                       : "cc",
                         "memory",
                         "v8",
                         "v9",
                         "v10",
                         "v11",
                         "v12",
                         "v13",
                         "v14",
                         "v15",
                         "v7",
                         "v17",
                         "v18",
                         "v19",
                         "v20",
                         "v21",
                         "v22",
                         "v24",
                         "v25",
                         "v26",
                         "v27",
                         "v28",
                         "v29",
                         "v30",
                         "v31");
#else   // not __aarch64__
          int cnt = w_loop;
          asm volatile(COMPUTE_C3
                       : [cnt] "+r"(cnt),
                         [r0] "+r"(inr0),
                         [r1] "+r"(inr1),
                         [r2] "+r"(inr2),
                         [wc] "+r"(wc0),
                         [ptr_out0] "+r"(pre_out0),
                         [wc0] "+r"(wc0)
                       :
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
#endif  // __aarch64__
          block_inr0 = block_inr2;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
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
      LITE_PARALLEL_END();
    }
  }
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
