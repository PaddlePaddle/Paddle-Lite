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

#include "lite/backends/arm/math/fp16/gemv_fp16.h"
#include <arm_neon.h>
#include "lite/core/parallel_defines.h"
namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
// clang-format off
#ifdef __aarch64__
#define GEMV_INIT                       \
  "prfm  pldl1keep, [%[ptr_in]] \n"     \
  "prfm  pldl1keep, [%[ptr_w0]] \n"     \
  "prfm  pldl1keep, [%[ptr_w1]] \n"     \
  "prfm  pldl1keep, [%[ptr_w2]] \n"     \
  "prfm  pldl1keep, [%[ptr_w3]] \n"     \
  "movi v9.8h, #0               \n"     \
  "prfm  pldl1keep, [%[ptr_w4]] \n"     \
  "movi v10.8h, #0               \n"    \
  "prfm  pldl1keep, [%[ptr_w5]] \n"     \
  "movi v11.8h, #0               \n"    \
  "prfm  pldl1keep, [%[ptr_w6]] \n"     \
  "movi v12.8h, #0               \n"    \
  "prfm  pldl1keep, [%[ptr_w7]] \n"     \
  "movi v13.8h, #0               \n"    \
  "cmp %w[cnt], #1              \n"     \
  "movi v14.8h, #0               \n"    \
  "movi v15.8h, #0               \n"    \
  "movi v16.8h, #0               \n"    \
  "blt 1f                       \n"     \
  "0:                           \n"     \
  "ld1 {v0.8h}, [%[ptr_in]], #16\n"     \
  "ld1 {v1.8h}, [%[ptr_w0]], #16\n"     \
  "ld1 {v2.8h}, [%[ptr_w1]], #16\n"     \
  "ld1 {v3.8h}, [%[ptr_w2]], #16\n"     \
  "ld1 {v4.8h}, [%[ptr_w3]], #16\n"     \
  "ld1 {v5.8h}, [%[ptr_w4]], #16\n"

#define GEMV_TRANS_COMPUTE              \
  "fmla v9.8h, v1.8h, v0.h[0]   \n"     \
  "ld1 {v6.8h}, [%[ptr_w5]], #16\n"     \
  "fmla v10.8h, v2.8h, v0.h[1]  \n"     \
  "ld1 {v7.8h}, [%[ptr_w6]], #16\n"     \
  "fmla v11.8h, v3.8h, v0.h[2]  \n"     \
  "ld1 {v8.8h}, [%[ptr_w7]], #16\n"     \
  "fmla v12.8h, v4.8h, v0.h[3]  \n"     \
  "add %[ptr_w0], %[ptr_w0], %[stride]\n"\
  "fmla v13.8h, v5.8h, v0.h[4]  \n"     \
  "add %[ptr_w1], %[ptr_w1], %[stride]\n"\
  "fmla v14.8h, v6.8h, v0.h[5]  \n"     \
  "add %[ptr_w2], %[ptr_w2], %[stride]\n"\
  "fmla v15.8h, v7.8h, v0.h[6]  \n"     \
  "add %[ptr_w3], %[ptr_w3], %[stride]\n"\
  "fmla v16.8h, v8.8h, v0.h[7]  \n"     \
  "add %[ptr_w4], %[ptr_w4], %[stride]\n"\
  "subs %w[cnt], %w[cnt], #1    \n"     \
  "add %[ptr_w5], %[ptr_w5], %[stride]\n"\
  "add %[ptr_w6], %[ptr_w6], %[stride]\n"\
  "add %[ptr_w7], %[ptr_w7], %[stride]\n"\
  "bne 0b                       \n"

#define GEMV_COMPUTE                     \
  "fmla v9.8h, v1.8h, v0.8h     \n"      \
  "ld1 {v6.8h}, [%[ptr_w5]], #16\n"      \
  "fmla v10.8h, v2.8h, v0.8h    \n"      \
  "ld1 {v7.8h}, [%[ptr_w6]], #16\n"      \
  "fmla v11.8h, v3.8h, v0.8h    \n"      \
  "ld1 {v8.8h}, [%[ptr_w7]], #16\n"      \
  "fmla v12.8h, v4.8h, v0.8h    \n"      \
  "ld1 {v17.8h}, [%[ptr_in]], #16\n"     \
  "fmla v13.8h, v5.8h, v0.8h    \n"      \
  "ld1 {v1.8h}, [%[ptr_w0]], #16\n"      \
  "fmla v14.8h, v6.8h, v0.8h    \n"      \
  "ld1 {v2.8h}, [%[ptr_w1]], #16\n"      \
  "fmla v15.8h, v7.8h, v0.8h    \n"      \
  "ld1 {v3.8h}, [%[ptr_w2]], #16\n"      \
  "fmla v16.8h, v8.8h, v0.8h    \n"      \
  "ld1 {v4.8h}, [%[ptr_w3]], #16\n"      \
  "fmla v9.8h, v1.8h, v17.8h    \n"      \
  "ld1 {v5.8h}, [%[ptr_w4]], #16\n"      \
  "fmla v10.8h, v2.8h, v17.8h   \n"      \
  "ld1 {v6.8h}, [%[ptr_w5]], #16\n"      \
  "fmla v11.8h, v3.8h, v17.8h   \n"      \
  "ld1 {v7.8h}, [%[ptr_w6]], #16\n"      \
  "fmla v12.8h, v4.8h, v17.8h   \n"      \
  "ld1 {v8.8h}, [%[ptr_w7]], #16\n"      \
  "fmla v13.8h, v5.8h, v17.8h   \n"      \
  "subs %w[cnt], %w[cnt], #1    \n"      \
  "fmla v14.8h, v6.8h, v17.8h   \n"      \
  "fmla v15.8h, v7.8h, v17.8h   \n"      \
  "fmla v16.8h, v8.8h, v17.8h   \n"      \
  "bne 0b                       \n"

#define STORE_TRANS                      \
  "1:                           \n"      \
  "cmp %w[flag_act], #0         \n"      \
  "fadd v0.8h, v9.8h, v10.8h    \n"      \
  "fadd v1.8h, v11.8h, v12.8h   \n"      \
  "fadd v2.8h, v13.8h, v14.8h   \n"      \
  "fadd v3.8h, v15.8h, v16.8h   \n"      \
  "beq 2f                       \n"      \
  "cmp %w[flag_act], #1         \n"      \
  "fadd v4.8h, v0.8h, v1.8h     \n"      \
  "fadd v5.8h, v2.8h, v3.8h    \n"       \
  "fadd v6.8h, v4.8h, v5.8h    \n"       \
  "fadd v6.8h, v6.8h, %[vbias].8h\n"     \
  "beq 3f                       \n"      \
  "cmp %w[flag_act], #2         \n"      \
  "beq 4f                       \n"      \
  "cmp %w[flag_act], #3         \n"      \
  "beq 6f                       \n"      \
  /* hardswish */                        \
  "fadd v7.8h, v6.8h, %[voffset].8h\n"   \
  "fmul v8.8h, v6.8h, %[valpha].8h\n"    \
  "fmax v7.8h, v7.8h, %[vzero].8h\n"     \
  "fmin v7.8h, v7.8h, %[vthreshold].8h\n"\
  "fmul v6.8h, v7.8h, v8.8h\n"           \
  "b 5f                         \n"      \
  /* leakyrelu */                        \
  "6:                             \n"    \
  "fmul v7.8h, v6.8h, %[valpha].8h\n"    \
  "fcmge v8.8h, v6.8h, %[vzero].8h\n"    \
  "bif  v6.16b, v7.16b, v8.16b  \n"      \
  "b 5f                         \n"      \
  /* relu6 */                            \
  "4:                           \n"      \
  "fmax v6.8h, v6.8h, %[vzero].8h\n"     \
  "fmin v6.8h, v6.8h, %[valpha].8h\n"    \
  "b 5f                         \n"      \
  /* relu */                             \
  "3:                           \n"      \
  "fmax v6.8h, v6.8h, %[vzero].8h\n"     \
  "b 5f                         \n"      \
  /* no act */                           \
  "2:                           \n"      \
  "fadd v4.8h, v0.8h, v1.8h    \n"       \
  "fadd v5.8h, v2.8h, v3.8h    \n"       \
  "fadd v6.8h, v4.8h, v5.8h    \n"       \
  "fadd v6.8h, v6.8h, %[vbias].8h\n"     \
  /* store */                            \
  "5:                           \n"      \
  "st1 {v6.8h}, [%[outptr]]     \n"

#define STORE                            \
  "1:                           \n"      \
  "cmp %w[flag_act], #0         \n"      \
  "faddp v0.8h, v9.8h, v10.8h   \n"      \
  "faddp v1.8h, v11.8h, v12.8h  \n"      \
  "faddp v2.8h, v13.8h, v14.8h  \n"      \
  "faddp v3.8h, v15.8h, v16.8h  \n"      \
  "beq 2f                       \n"      \
  "cmp %w[flag_act], #1         \n"      \
  "faddp v4.8h, v0.8h, v1.8h    \n"      \
  "faddp v5.8h, v2.8h, v3.8h    \n"      \
  "faddp v6.8h, v4.8h, v5.8h    \n"      \
  "fadd v6.8h, v6.8h, %[vbias].8h\n"     \
  "beq 3f                       \n"      \
  "cmp %w[flag_act], #2         \n"      \
  "beq 4f                       \n"      \
  "cmp %w[flag_act], #3         \n"      \
  "beq 6f                       \n"      \
  /* hardswish */                        \
  "fadd v7.8h, v6.8h, %[voffset].8h\n"   \
  "fmul v8.8h, v6.8h, %[valpha].8h\n"    \
  "fmax v7.8h, v7.8h, %[vzero].8h\n"     \
  "fmin v7.8h, v7.8h, %[vthreshold].8h\n"\
  "fmul v6.8h, v7.8h, v8.8h\n"           \
  "b 5f                         \n"      \
  /* leakyrelu */                        \
  "6:                             \n"    \
  "fmul v7.8h, v6.8h, %[valpha].8h\n"    \
  "fcmge v8.8h, v6.8h, %[vzero].8h\n"    \
  "bif  v6.16b, v7.16b, v8.16b  \n"      \
  "b 5f                         \n"      \
  /* relu6 */                            \
  "4:                           \n"      \
  "fmax v6.8h, v6.8h, %[vzero].8h\n"     \
  "fmin v6.8h, v6.8h, %[valpha].8h\n"    \
  "b 5f                         \n"      \
  /* relu */                             \
  "3:                           \n"      \
  "fmax v6.8h, v6.8h, %[vzero].8h\n"     \
  "b 5f                         \n"      \
  /* no act */                           \
  "2:                           \n"      \
  "faddp v4.8h, v0.8h, v1.8h    \n"      \
  "faddp v5.8h, v2.8h, v3.8h    \n"      \
  "faddp v6.8h, v4.8h, v5.8h    \n"      \
  "fadd v6.8h, v6.8h, %[vbias].8h\n"     \
  /* store */                            \
  "5:                           \n"      \
  "st1 {v6.8h}, [%[outptr]]     \n"

#define GEMV_ASM_PARAMS                  \
      [ptr_in] "+r"(ptr_in),             \
      [ptr_w0] "+r"(ptr_w0),             \
      [ptr_w1] "+r"(ptr_w1),             \
      [ptr_w2] "+r"(ptr_w2),             \
      [ptr_w3] "+r"(ptr_w3),             \
      [ptr_w4] "+r"(ptr_w4),             \
      [ptr_w5] "+r"(ptr_w5),             \
      [ptr_w6] "+r"(ptr_w6),             \
      [ptr_w7] "+r"(ptr_w7),             \
      [cnt] "+r"(cnt_col)                \
    : [vbias] "w"(vbias),                \
      [vzero] "w"(vzero),                \
      [valpha] "w"(valpha),              \
      [voffset] "w"(voffset),            \
      [vthreshold] "w"(vthreshold),      \
      [flag_act] "r"(flag_act),          \
      [outptr] "r"(out_p),              \
      [stride] "r"(stride)               \
    : "cc", "memory", "v0", "v1", "v2",  \
      "v3", "v4", "v5", "v6", "v7",      \
      "v8", "v9", "v10", "v11", "v12",   \
      "v13", "v14", "v15", "v16", "v17"

#else
#define GEMV_INIT                       \
  "pld    [%[ptr_in]]           \n"     \
  "pld    [%[ptr_w0]]           \n"     \
  "pld    [%[ptr_w1]]           \n"     \
  "pld    [%[ptr_w2]]           \n"     \
  "pld    [%[ptr_w3]]           \n"     \
  "vmov.u32 q12, #0             \n"     \
  "vmov.u32 q13, #0             \n"     \
  "vmov.u32 q14, #0             \n"     \
  "vmov.u32 q15, #0             \n"

#define GEMV_TRANS_COMPUTE              \
  "cmp  %[cnt], #8              \n"     \
  "blt 1f                       \n"     \
  "0:                           \n"     \
  "vld1.16 {d0-d1},   [%[ptr_in]]!\n"   \
  "vld1.16 {d2-d3},   [%[ptr_w0]]!\n"   \
  "vmov.u32 q8,  #0             \n"     \
  "vld1.16 {d4-d5},   [%[ptr_w1]]!\n"   \
  "vmov.u32 q9,  #0             \n"     \
  "vld1.16 {d6-d7},   [%[ptr_w2]]!\n"   \
  "vmov.u32 q10, #0             \n"     \
  "vld1.16 {d8-d9},   [%[ptr_w3]]!\n"   \
  "vmov.u32 q11, #0             \n"     \
  "vld1.16 {d10-d11},   [%[ptr_w4]]!\n" \
  "vmla.f16 q8,  q1, d0[0]      \n"     \
  "vld1.16 {d12-d13},   [%[ptr_w5]]!\n" \
  "vmla.f16 q9,  q2, d0[1]      \n"     \
  "vld1.16 {d14-d15},   [%[ptr_w6]]!\n" \
  "vmla.f16 q10, q3, d0[2]      \n"     \
  "vld1.16 {d2-d3},     [%[ptr_w7]]!\n" \
  "vmla.f16 q11, q4, d0[3]      \n"     \
  "add %[ptr_w0],  %[stride]    \n"     \
  "vmla.f16 q12, q5, d1[0]      \n"     \
  "add %[ptr_w1],  %[stride]    \n"     \
  "vmla.f16 q13, q6, d1[1]      \n"     \
  "add %[ptr_w2],  %[stride]    \n"     \
  "vmla.f16 q14, q7, d1[2]      \n"     \
  "add %[ptr_w3],  %[stride]    \n"     \
  "vmla.f16 q15, q1, d1[3]      \n"     \
  "subs %[cnt],  #8             \n"     \
  "add %[ptr_w4],  %[stride]    \n"     \
  "add %[ptr_w5],  %[stride]    \n"     \
  "add %[ptr_w6],  %[stride]    \n"     \
  "add %[ptr_w7],  %[stride]    \n"     \
  "bne 0b                       \n"

#define GEMV_COMPUTE                     \
  "cmp  %[cnt], #1              \n"      \
  "blt 1f                       \n"      \
  "0:                           \n"      \
  "vld1.16 {d8-d9},   [%[ptr_in]]!\n"    \
  "vld1.16 {d12-d13}, [%[ptr_w0]]!\n"    \
  "vld1.16 {d14-d15}, [%[ptr_w1]]!\n"    \
  "vld1.16 {d16-d17}, [%[ptr_w2]]!\n"    \
  "vld1.16 {d18-d19}, [%[ptr_w3]]!\n"    \
  "vld1.16 {d10-d11}, [%[ptr_in]]!\n"    \
  "vmla.f16 q12,  q6,  q4       \n"      \
  "vld1.16 {d12-d13}, [%[ptr_w0]]!\n"    \
  "vmla.f16 q13,  q7,  q4       \n"      \
  "vld1.16 {d14-d15}, [%[ptr_w1]]!\n"    \
  "vmla.f16 q14,  q8,  q4       \n"      \
  "vld1.16 {d16-d17}, [%[ptr_w2]]!\n"    \
  "vmla.f16 q15,  q9,  q4       \n"      \
  "vld1.16 {d18-d19}, [%[ptr_w3]]!\n"    \
  "subs    %[cnt], #1           \n"      \
  "vmla.f16 q12,  q6,  q5       \n"      \
  "vmla.f16 q13,  q7,  q5       \n"      \
  "vmla.f16 q14,  q8,  q5       \n"      \
  "vmla.f16 q15,  q9,  q5       \n"      \
  "bne 0b                       \n"

#define STORE_TRANS                      \
  "1:                           \n"      \
  "cmp  %[cnt], #0              \n"      \
  "vadd.f16 q2, q8,  q9         \n"      \
  "vadd.f16 q3, q10, q11        \n"      \
  "vadd.f16 q4, q12, q13        \n"      \
  "vadd.f16 q5, q14, q15        \n"      \
  "beq 2f                       \n"      \
  "cmp  %[cnt], #1              \n"      \
  "vldr   d0, [%[alpha_ptr], #48]\n"     \
  "vldr   d1, [%[alpha_ptr], #56]\n"     \
  "vadd.f16 q10, q2,  q3        \n"      \
  "vadd.f16 q11, q4,  q5        \n"      \
  "vmov.u32 q15, #0             \n"      \
  "vadd.f16 q12, q10, q11       \n"      \
  "beq 3f                       \n"      \
  "cmp  %[cnt], #2              \n"      \
  "beq 4f                       \n"      \
  "cmp  %[cnt], #3              \n"      \
  "beq 6f                       \n"      \
  /* hardswish */                        \
  "vld1.16 {d2-d5}, [%[alpha_ptr]]\n"    \
  "vadd.f16 q12, q12, q0        \n"      \
  "vldr   d0, [%[alpha_ptr], #32]\n"     \
  "vldr   d1, [%[alpha_ptr], #40]\n"     \
  "vadd.f16 q13, q12, q2        \n"      \
  "vmul.f16 q14, q12, q1        \n"      \
  "vmax.f16 q13, q13, q15       \n"      \
  "vmin.f16 q13, q13, q0        \n"      \
  "vmul.f16 q12, q13, q14       \n"      \
  "b 5f                         \n"      \
  /* leakyrelu */                        \
  "6:                             \n"    \
  "vld1.16 {d2-d3}, [%[alpha_ptr]]\n"    \
  "vadd.f16 q12, q12, q0        \n"      \
  "vmul.f16 q13, q12, q1        \n"      \
  "vcge.f16 q14, q12, q15       \n"      \
  "vbif     q12, q13, q14       \n"      \
  "b 5f                         \n"      \
  /* relu6 */                            \
  "4:                           \n"      \
  "vld1.16 {d2-d3}, [%[alpha_ptr]]\n"    \
  "vadd.f16 q12, q12, q0        \n"      \
  "vmax.f16 q12, q12, q15       \n"      \
  "vmin.f16 q12, q12, q1        \n"      \
  "b 5f                         \n"      \
  /* relu */                             \
  "3:                           \n"      \
  "vld1.16 {d2-d3}, [%[alpha_ptr]]\n"    \
  "vadd.f16 q12, q12, q0        \n"      \
  "vmax.f16 q12, q12, q15       \n"      \
  "b 5f                         \n"      \
  /* no act */                           \
  "2:                           \n"      \
  "vldr   d0, [%[alpha_ptr], #48]\n"     \
  "vldr   d1, [%[alpha_ptr], #56]\n"     \
  "vadd.f16 q10, q2,  q3        \n"      \
  "vadd.f16 q11, q4,  q5        \n"      \
  "vmov.u32 q15, #0             \n"      \
  "vadd.f16 q12, q10, q11       \n"      \
  "vadd.f16 q12, q12, q0        \n"      \
  /* store */                            \
  "5:                           \n"      \
  "vst1.16 {d24-d25}, [%[outptr]]   \n"

#define STORE                            \
  "1:                           \n"      \
  "cmp  %[flag_act], #0         \n"      \
  "vpadd.f16 d8,  d24, d25      \n"      \
  "vpadd.f16 d9,  d26, d27      \n"      \
  "vpadd.f16 d10, d28, d29      \n"      \
  "vpadd.f16 d11, d30, d31      \n"      \
  "beq 2f                       \n"      \
  "cmp  %[flag_act], #1         \n"      \
  "vpadd.f16 d20, d8,  d9       \n"      \
  "vpadd.f16 d21, d10, d11      \n"      \
  "vpadd.f16 d12, d20, d21      \n"      \
  "vadd.f16  d12, d12, %e[vbias]\n"      \
  "beq 3f                       \n"      \
  "cmp  %[flag_act], #2         \n"      \
  "beq 4f                       \n"      \
  "cmp  %[flag_act], #3         \n"      \
  "beq 6f                       \n"      \
  /* hardswish */                        \
  "vld1.16 {d8-d9}, [%[hard_parameter]]\n"\
  "vmul.f16 d16, d12, %e[valpha]\n"      \
  "vadd.f16 d14, d12, d8        \n"      \
  "vmax.f16 d14, d14, %e[vzero] \n"      \
  "vmin.f16 d14, d14, d9        \n"      \
  "vmul.f16 d12, d14, d16       \n"      \
  "b 5f                         \n"      \
  /* leakyrelu */                        \
  "6:                             \n"    \
  "vmul.f16 d16, d12, %e[valpha]\n"      \
  "vcge.f16 d14, d12, %e[vzero] \n"      \
  "vbif     d12, d16, d14       \n"      \
  "b 5f                         \n"      \
  /* relu6 */                            \
  "4:                           \n"      \
  "vmax.f16 d12, d12, %e[vzero] \n"      \
  "vmin.f16 d12, d12, %e[valpha]\n"      \
  "b 5f                         \n"      \
  /* relu */                             \
  "3:                           \n"      \
  "vmax.f16 d12, d12, %e[vzero] \n"      \
  "b 5f                         \n"      \
  /* no act */                           \
  "2:                           \n"      \
  "vpadd.f16 d20, d8,  d9       \n"      \
  "vpadd.f16 d21, d10, d11      \n"      \
  "vpadd.f16 d12, d20, d21      \n"      \
  "vadd.f16 d12,  d12, %e[vbias]\n"      \
  /* store */                            \
  "5:                           \n"      \
  "vst1.16 {d12}, [%[outptr]]   \n"

#define GEMV_ASM_PARAMS                  \
      [ptr_in] "+r"(ptr_in),             \
      [ptr_w0] "+r"(ptr_w0),             \
      [ptr_w1] "+r"(ptr_w1),             \
      [ptr_w2] "+r"(ptr_w2),             \
      [ptr_w3] "+r"(ptr_w3),             \
      [cnt] "+r"(cnt_col)                \
    : [vbias] "w"(vbias),                \
      [vzero] "w"(vzero),                \
      [valpha] "w"(valpha),              \
      [flag_act] "r"(flag_act),          \
      [hard_parameter] "r"(hard_parameter), \
      [outptr] "r"(out_p),               \
      [stride] "r"(stride)               \
    : "cc", "memory", "q4", "q5", "q6",  \
      "q7", "q8", "q9", "q10", "q11",    \
      "q12", "q13", "q14", "q15"

#define GEMV_ASM_PARAMS_TRANS            \
      [ptr_in] "+r"(ptr_in),             \
      [ptr_w0] "+r"(ptr_w0),             \
      [ptr_w1] "+r"(ptr_w1),             \
      [ptr_w2] "+r"(ptr_w2),             \
      [ptr_w3] "+r"(ptr_w3),             \
      [ptr_w4] "+r"(ptr_w4),             \
      [ptr_w5] "+r"(ptr_w5),             \
      [ptr_w6] "+r"(ptr_w6),             \
      [ptr_w7] "+r"(ptr_w7),             \
      [cnt] "+r"(cnt_col)                \
    : [alpha_ptr] "r"(alpha_ptr),        \
      [outptr] "r"(out_p),               \
      [stride] "r"(stride)               \
    : "cc", "memory", "q0", "q1", "q2", "q3", \
      "q4", "q5", "q6", "q7", "q8", "q9", \
      "q10", "q11", "q12", "q13", "q14", "q15"
#endif
// clang-format on
void gemv_fp16_trans(const float16_t *A,
                     const float16_t *x,
                     float16_t *y,
                     int M,
                     int N,
                     float16_t beta,
                     bool is_bias,
                     const float16_t *bias,
                     bool is_act,
                     const operators::ActivationParam act_param,
                     ARMContext *ctx) {
  int Nup = (N + 7) / 8 * 8;
  int Mup = (M + 7) / 8 * 8;
  auto ptr_zero = ctx->workspace_data<float16_t>();
  memset(ptr_zero, 0, Nup * sizeof(float16_t));
  auto bias_ptr = ptr_zero + Nup;
  if (is_bias) {
    lite::TargetWrapperHost::MemcpySync(bias_ptr, bias, M * sizeof(float16_t));
  } else {
    memset(bias_ptr, 0, Mup * sizeof(float16_t));
  }
  float16_t *ptr_w = bias_ptr + Mup;
  lite::TargetWrapperHost::MemcpySync(ptr_w, A, N * sizeof(float16_t));
  memset(ptr_w + N, 0, (Nup - N) * sizeof(float16_t));
  float16_t *data_in = ptr_w + Nup;
  lite::TargetWrapperHost::MemcpySync(
      data_in, x + (M - 1) * N, N * sizeof(float16_t));
  int cnt = Nup >> 3;
  float16_t local_alpha = 0.f;
  float16_t offset = 0.f;
  float16_t threshold = 6.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (is_act) {
    act_acquire(act_param.active_type,
                flag_act,
                local_alpha,
                offset,
                threshold,
                act_param);
  }
#ifdef __aarch64__
  int out_cnt = M >> 3;
  int remain = M & 7;
  if (remain > 0) out_cnt++;
  float16x8_t vzero = vdupq_n_f16(0.f);
  float16x8_t valpha = vdupq_n_f16(local_alpha);
  float16x8_t voffset = vdupq_n_f16(offset);
  float16x8_t vthreshold = vdupq_n_f16(threshold);
  int stride = 16 * (M - 1);  // (8 * M - 8) * 2
  int rem_n = N & 7;
  LITE_PARALLEL_BEGIN(j, tid, out_cnt) {
    int out_idx = j * 8;
    float16_t out_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float16_t *out_ptr = y + out_idx;
    float16_t *out_p = out_ptr;
    const float16_t *ptr_in = ptr_w;
    const float16_t *ptr_w0 = x + out_idx;
    const float16_t *ptr_w1 = ptr_w0 + M;
    const float16_t *ptr_w2 = ptr_w1 + M;
    const float16_t *ptr_w3 = ptr_w2 + M;
    const float16_t *ptr_w4 = ptr_w3 + M;
    const float16_t *ptr_w5 = ptr_w4 + M;
    const float16_t *ptr_w6 = ptr_w5 + M;
    const float16_t *ptr_w7 = ptr_w6 + M;
    float16x8_t vbias = vld1q_f16(bias_ptr + out_idx);
    ptr_acquire_norm<float16_t>(ptr_zero,
                                ptr_w0,
                                ptr_w1,
                                ptr_w2,
                                ptr_w3,
                                ptr_w4,
                                ptr_w5,
                                ptr_w6,
                                ptr_w7,
                                rem_n);
    if (j == out_cnt - 1 && remain) {
      ptr_acquire_remain<float16_t>(data_in,
                                    ptr_w0,
                                    ptr_w1,
                                    ptr_w2,
                                    ptr_w3,
                                    ptr_w4,
                                    ptr_w5,
                                    ptr_w6,
                                    ptr_w7,
                                    remain);
      out_p = out_temp;
    }
    // 8x8
    int cnt_col = cnt;
    asm volatile(GEMV_INIT GEMV_TRANS_COMPUTE STORE_TRANS : GEMV_ASM_PARAMS);
    if (remain > 0) {
      for (int i = 0; i < remain; i++) {
        out_ptr[i] = out_p[i];
      }
    }
  }
  LITE_PARALLEL_END();
#else
  int out_cnt = M >> 3;
  int remain = M & 7;
  if (remain > 0) out_cnt++;
  float16_t alpha_ptr[32] = {0.f};
  for (int i = 0; i < 8; i++) {
    alpha_ptr[i] = local_alpha;
    alpha_ptr[i * 8] = offset;
    alpha_ptr[i * 16] = threshold;
  }
  int stride = 16 * (M - 1);  // (8 * M - 8) * 2
  int rem_n = N & 7;
  int cnt_act = cnt * 8 + flag_act;
  LITE_PARALLEL_BEGIN(j, tid, out_cnt) {
    int out_idx = j * 8;
    float16_t out_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float16_t *out_ptr = y + out_idx;
    float16_t *out_p = out_ptr;
    const float16_t *ptr_in = ptr_w;
    const float16_t *ptr_w0 = x + out_idx;
    const float16_t *ptr_w1 = ptr_w0 + M;
    const float16_t *ptr_w2 = ptr_w1 + M;
    const float16_t *ptr_w3 = ptr_w2 + M;
    const float16_t *ptr_w4 = ptr_w3 + M;
    const float16_t *ptr_w5 = ptr_w4 + M;
    const float16_t *ptr_w6 = ptr_w5 + M;
    const float16_t *ptr_w7 = ptr_w6 + M;
    for (int i = 0; i < 8; i++) {
      alpha_ptr[32 + i] = bias_ptr[out_idx + i];
    }
    ptr_acquire_norm<float16_t>(ptr_zero,
                                ptr_w0,
                                ptr_w1,
                                ptr_w2,
                                ptr_w3,
                                ptr_w4,
                                ptr_w5,
                                ptr_w6,
                                ptr_w7,
                                rem_n);
    if (j == out_cnt - 1 && remain) {
      ptr_acquire_remain<float16_t>(data_in,
                                    ptr_w0,
                                    ptr_w1,
                                    ptr_w2,
                                    ptr_w3,
                                    ptr_w4,
                                    ptr_w5,
                                    ptr_w6,
                                    ptr_w7,
                                    remain);
      out_p = out_temp;
    }
    // 8x8
    int cnt_col = cnt_act;
    asm volatile(GEMV_INIT GEMV_TRANS_COMPUTE STORE_TRANS
                 : GEMV_ASM_PARAMS_TRANS);
    if (remain > 0) {
      for (int i = 0; i < remain; i++) {
        out_ptr[i] = out_p[i];
      }
    }
  }
  LITE_PARALLEL_END();
#endif
}

void gemv_fp16(const float16_t *A,
               const float16_t *x,
               float16_t *y,
               bool transA,
               int M,
               int N,
               float16_t beta,
               bool is_bias,
               const float16_t *bias,
               bool is_act,
               const operators::ActivationParam act_param,
               ARMContext *ctx) {
  if (transA) {
    // 8x16
    gemv_fp16_trans(A, x, y, M, N, beta, is_bias, bias, is_act, act_param, ctx);
    return;
  }
  int Nup = (N + 15) / 16 * 16;
  int Mup = (M + 7) / 8 * 8;
  auto ptr_zero = ctx->workspace_data<float16_t>();
  memset(ptr_zero, 0, Nup * sizeof(float16_t));
  auto bias_ptr = ptr_zero + Nup;
  if (is_bias) {
    lite::TargetWrapperHost::MemcpySync(bias_ptr, bias, M * sizeof(float16_t));
    memset(bias_ptr + M, 0, (Mup - M) * sizeof(float16_t));
  } else {
    memset(bias_ptr, 0, Mup * sizeof(float16_t));
  }

  float16_t *data_in = bias_ptr + Mup;
  lite::TargetWrapperHost::MemcpySync(data_in, x, N * sizeof(float16_t));
  memset(data_in + N, 0, (Nup - N) * sizeof(float16_t));
  float16_t *ptr_w = data_in + Nup;
  lite::TargetWrapperHost::MemcpySync(
      ptr_w, A + (M - 1) * N, N * sizeof(float16_t));
  int cnt = Nup >> 4;
  float16_t local_alpha = 0.f;
  float16_t offset = 0.f;
  float16_t threshold = 6.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (is_act) {
    act_acquire(act_param.active_type,
                flag_act,
                local_alpha,
                offset,
                threshold,
                act_param);
  }

  float16x8_t vzero = vdupq_n_f16(0.f);
  float16x8_t valpha = vdupq_n_f16(local_alpha);
#ifdef __aarch64__
  int out_cnt = M >> 3;
  int remain = M & 7;
  if (remain > 0) out_cnt++;
  float16x8_t voffset = vdupq_n_f16(offset);
  float16x8_t vthreshold = vdupq_n_f16(threshold);
  int stride = 1;

  LITE_PARALLEL_BEGIN(j, tid, out_cnt) {
    int out_idx = j * 8;
    float16_t out_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float16_t *out_ptr = y + out_idx;
    float16_t *out_p = out_ptr;
    const float16_t *ptr_in = data_in;
    const float16_t *ptr_w0 = A + (N * out_idx);
    const float16_t *ptr_w1 = ptr_w0 + N;
    const float16_t *ptr_w2 = ptr_w1 + N;
    const float16_t *ptr_w3 = ptr_w2 + N;
    const float16_t *ptr_w4 = ptr_w3 + N;
    const float16_t *ptr_w5 = ptr_w4 + N;
    const float16_t *ptr_w6 = ptr_w5 + N;
    const float16_t *ptr_w7 = ptr_w6 + N;
    float16x8_t vbias = vld1q_f16(bias_ptr + out_idx);
    if (j == out_cnt - 1 && remain) {
      ptr_acquire_norm<float16_t>(ptr_zero,
                                  ptr_w0,
                                  ptr_w1,
                                  ptr_w2,
                                  ptr_w3,
                                  ptr_w4,
                                  ptr_w5,
                                  ptr_w6,
                                  ptr_w7,
                                  remain);
      out_p = out_temp;
      ptr_acquire_remain<float16_t>(ptr_w,
                                    ptr_w0,
                                    ptr_w1,
                                    ptr_w2,
                                    ptr_w3,
                                    ptr_w4,
                                    ptr_w5,
                                    ptr_w6,
                                    ptr_w7,
                                    remain);
    }
    // 8x16
    int cnt_col = cnt;
    asm volatile(GEMV_INIT GEMV_COMPUTE STORE : GEMV_ASM_PARAMS);
    if (remain > 0) {
      for (int i = 0; i < remain; i++) {
        out_ptr[i] = out_p[i];
      }
    }
  }
  LITE_PARALLEL_END();
#else
  int out_cnt = M >> 2;
  int remain = M & 3;
  if (remain > 0) out_cnt++;
  float16_t hard_parameter[8] = {offset,
                                 offset,
                                 offset,
                                 offset,
                                 threshold,
                                 threshold,
                                 threshold,
                                 threshold};
  int stride = 1;

  LITE_PARALLEL_BEGIN(j, tid, out_cnt) {
    int out_idx = j * 4;
    float16_t out_temp[4] = {0, 0, 0, 0};
    float16_t *out_ptr = y + out_idx;
    float16_t *out_p = out_ptr;
    const float16_t *ptr_in = data_in;
    const float16_t *ptr_w0 = A + (N * out_idx);
    const float16_t *ptr_w1 = ptr_w0 + N;
    const float16_t *ptr_w2 = ptr_w1 + N;
    const float16_t *ptr_w3 = ptr_w2 + N;
    float16x8_t vbias = vld1q_f16(bias_ptr + out_idx);
    if (j == out_cnt - 1 && remain) {
      ptr_acquire_norm_four<float16_t>(
          ptr_zero, ptr_w0, ptr_w1, ptr_w2, ptr_w3, remain);
      out_p = out_temp;
      ptr_acquire_remain_four<float16_t>(
          ptr_w, ptr_w0, ptr_w1, ptr_w2, ptr_w3, remain);
    }
    // 4x16
    int cnt_col = cnt;
    asm volatile(GEMV_INIT GEMV_COMPUTE STORE : GEMV_ASM_PARAMS);
    if (remain > 0) {
      for (int i = 0; i < remain; i++) {
        out_ptr[i] = out_p[i];
      }
    }
  }
  LITE_PARALLEL_END();
#endif
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
