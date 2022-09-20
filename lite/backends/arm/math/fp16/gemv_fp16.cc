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
  auto size = (Mup * 2 + Nup);
  ctx->ExtendWorkspace(size * sizeof(float16_t));
  auto ptr_zero = ctx->workspace_data<float16_t>();
  memset(ptr_zero, 0, Mup * sizeof(float16_t));
  auto bias_ptr = ptr_zero + Mup;
  if (is_bias) {
    lite::TargetWrapperHost::MemcpySync(bias_ptr, bias, M * sizeof(float16_t));
    memset(bias_ptr + M, 0, (Mup - M) * sizeof(float16_t));
  } else {
    memset(bias_ptr, 0, Mup * sizeof(float16_t));
  }
  float16_t *ptr_w = bias_ptr + Mup;
  lite::TargetWrapperHost::MemcpySync(ptr_w, x, N * sizeof(float16_t));
  memset(ptr_w + N, 0, (Nup - N) * sizeof(float16_t));
  memset(y, 0, M * sizeof(float16_t));
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
  int out_cnt = M >> 3;
  int remain = M & 7;
  int cnt_n = N >> 3;
  int rem_n = N & 7;
  if (rem_n > 0) cnt_n++;
  LITE_PARALLEL_BEGIN(j, tid, cnt_n) {
    int y_index = j * 8;
    const float16_t *ptr_in = ptr_w + y_index;
    const float16_t *inptr_row[8];
    inptr_row[0] = A + y_index * M;
    for (int i = 1; i < 8; i++) {
      inptr_row[i] = inptr_row[i - 1] + M;
    }
    float16_t *out_ptr = y;
    if (j == cnt_n - 1 && rem_n) {
      ptr_acquire_norm<float16_t>(ptr_zero,
                                  &inptr_row[0],
                                  &inptr_row[1],
                                  &inptr_row[2],
                                  &inptr_row[3],
                                  &inptr_row[4],
                                  &inptr_row[5],
                                  &inptr_row[6],
                                  &inptr_row[7],
                                  rem_n);
    }
    for (int i = 0; i < out_cnt; i++) {
// 1x8 + 8x8 = 1x8
#ifdef __aarch64__
      asm volatile(
          "ld1 {v0.8h}, [%[ptr_in]]\n"
          "ld1 {v9.8h}, [%[out_ptr]]    \n"
          "ld1 {v1.8h}, [%[ptr_w0]], #16\n"
          "ld1 {v2.8h}, [%[ptr_w1]], #16\n"
          "ld1 {v3.8h}, [%[ptr_w2]], #16\n"
          "ld1 {v4.8h}, [%[ptr_w3]], #16\n"
          "ld1 {v5.8h}, [%[ptr_w4]], #16\n"
          "fmla v9.8h,  v1.8h, v0.h[0]   \n"
          "ld1 {v6.8h}, [%[ptr_w5]], #16\n"
          "fmul v10.8h, v2.8h, v0.h[1]   \n"
          "ld1 {v7.8h}, [%[ptr_w6]], #16\n"
          "fmul v11.8h, v3.8h, v0.h[2]   \n"
          "ld1 {v8.8h}, [%[ptr_w7]], #16\n"
          "fmul v12.8h, v4.8h, v0.h[3]   \n"
          "fmla v9.8h,  v5.8h, v0.h[4]   \n"
          "fmla v10.8h, v6.8h, v0.h[5]   \n"
          "fmla v11.8h, v7.8h, v0.h[6]   \n"
          "fmla v12.8h, v8.8h, v0.h[7]   \n"
          "fadd v0.8h,  v9.8h,  v10.8h   \n"
          "fadd v1.8h,  v11.8h, v12.8h   \n"
          "fadd v2.8h,  v0.8h,  v1.8h    \n"
          "st1  {v2.8h}, [%[out_ptr]]    \n"
          : [ptr_w0] "+r"(inptr_row[0]),
            [ptr_w1] "+r"(inptr_row[1]),
            [ptr_w2] "+r"(inptr_row[2]),
            [ptr_w3] "+r"(inptr_row[3]),
            [ptr_w4] "+r"(inptr_row[4]),
            [ptr_w5] "+r"(inptr_row[5]),
            [ptr_w6] "+r"(inptr_row[6]),
            [ptr_w7] "+r"(inptr_row[7])
          : [ptr_in] "r"(ptr_in), [out_ptr] "r"(out_ptr)
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
            "v12");
#else
      asm volatile(
          "vld1.16 {d0-d1},   [%[ptr_in]]\n"
          "vld1.16 {d18-d19}, [%[out_ptr]]\n"
          "vld1.16 {d2-d3},   [%[ptr_w0]]! \n"
          "vld1.16 {d4-d5},   [%[ptr_w1]]!\n"
          "vld1.16 {d6-d7},   [%[ptr_w2]]!\n"
          "vld1.16 {d8-d9},   [%[ptr_w3]]!\n"
          "vld1.16 {d10-d11}, [%[ptr_w4]]!\n"
          "vmla.f16 q9,  q1,  d0[0]      \n"
          "vld1.16 {d12-d13}, [%[ptr_w5]]!\n"
          "vmul.f16 q10, q2,  d0[1]      \n"
          "vld1.16 {d14-d15}, [%[ptr_w6]]!\n"
          "vmul.f16 q11, q3,  d0[2]      \n"
          "vld1.16 {d2-d3},   [%[ptr_w7]]! \n"
          "vmul.f16 q12, q4,  d0[3]      \n"
          "vmla.f16 q9,  q5,  d1[0]      \n"
          "vmla.f16 q10, q6,  d1[1]      \n"
          "vmla.f16 q11, q7,  d1[2]      \n"
          "vmla.f16 q12, q1,  d1[3]      \n"
          "vadd.f16 q0,  q9,  q10        \n"
          "vadd.f16 q1,  q11, q12        \n"
          "vadd.f16 q2,  q0,  q1         \n"
          "vst1.16  {d4-d5}, [%[out_ptr]]    \n"
          : [ptr_w0] "+r"(inptr_row[0]),
            [ptr_w1] "+r"(inptr_row[1]),
            [ptr_w2] "+r"(inptr_row[2]),
            [ptr_w3] "+r"(inptr_row[3]),
            [ptr_w4] "+r"(inptr_row[4]),
            [ptr_w5] "+r"(inptr_row[5]),
            [ptr_w6] "+r"(inptr_row[6]),
            [ptr_w7] "+r"(inptr_row[7])
          : [ptr_in] "r"(ptr_in), [out_ptr] "r"(out_ptr)
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
            "q12");
#endif
      if (j == cnt_n - 1) {
        for (int k = 0; k < 8; k++) {
          out_ptr[k] += bias_ptr[i * 8 + k];
          if (flag_act == 1) {
            out_ptr[k] = out_ptr[k] > 0.f ? out_ptr[k] : 0.f;
          } else if (flag_act == 2) {
            out_ptr[k] =
                out_ptr[k] > 0.f
                    ? (out_ptr[k] < local_alpha ? out_ptr[k] : local_alpha)
                    : 0.f;
          } else if (flag_act == 3) {
            out_ptr[k] =
                out_ptr[k] > 0.f ? out_ptr[k] : out_ptr[k] * local_alpha;
          } else if (flag_act == 4) {
            auto tmp0 = out_ptr[k] + offset;
            auto tmp1 = out_ptr[k] * local_alpha;
            tmp0 = tmp0 > 0.f ? (tmp0 < threshold ? tmp0 : threshold) : 0.f;
            out_ptr[k] = tmp0 * tmp1;
          }
        }
      }
      out_ptr += 8;
    }
    for (int i = 0; i < remain; i++) {
      float16_t sum = 0.f;
      for (int k = 0; k < 8; k++) {
        sum += (*inptr_row[k]) * ptr_in[k];
        inptr_row[k]++;
      }
      if (j == cnt_n - 1) {
        sum += bias_ptr[out_cnt * 8 + i];
        *out_ptr += sum;
        if (flag_act == 1) {
          out_ptr[0] = out_ptr[0] > 0.f ? out_ptr[0] : 0.f;
        } else if (flag_act == 2) {
          out_ptr[0] =
              out_ptr[0] > 0.f
                  ? (out_ptr[0] < local_alpha ? out_ptr[0] : local_alpha)
                  : 0.f;
        } else if (flag_act == 3) {
          out_ptr[0] = out_ptr[0] > 0.f ? out_ptr[0] : out_ptr[0] * local_alpha;
        } else if (flag_act == 4) {
          auto tmp0 = out_ptr[0] + offset;
          auto tmp1 = out_ptr[0] * local_alpha;
          tmp0 = tmp0 > 0.f ? (tmp0 < threshold ? tmp0 : threshold) : 0.f;
          out_ptr[0] = tmp0 * tmp1;
        }
      } else {
        *out_ptr += sum;
      }
      out_ptr++;
    }
  }
  LITE_PARALLEL_END();
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
  auto size = (Nup * 3 + Mup);
  ctx->ExtendWorkspace(size * sizeof(float16_t));
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
  memset(ptr_w + N, 0, (Nup - N) * sizeof(float16_t));
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
                                  &ptr_w0,
                                  &ptr_w1,
                                  &ptr_w2,
                                  &ptr_w3,
                                  &ptr_w4,
                                  &ptr_w5,
                                  &ptr_w6,
                                  &ptr_w7,
                                  remain);
      out_p = out_temp;
      ptr_acquire_remain<float16_t>(ptr_w,
                                    &ptr_w0,
                                    &ptr_w1,
                                    &ptr_w2,
                                    &ptr_w3,
                                    &ptr_w4,
                                    &ptr_w5,
                                    &ptr_w6,
                                    &ptr_w7,
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
  float16_t hard_parameter[8];
  for (int i = 0; i < 4; i++) {
    hard_parameter[i] = offset;
    hard_parameter[i + 4] = threshold;
  }
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
          ptr_zero, &ptr_w0, &ptr_w1, &ptr_w2, &ptr_w3, remain);
      out_p = out_temp;
      ptr_acquire_remain_four<float16_t>(
          ptr_w, &ptr_w0, &ptr_w1, &ptr_w2, &ptr_w3, remain);
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
